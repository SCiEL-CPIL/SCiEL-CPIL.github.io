"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Build coverage map: for each word in the space, which codewords cover it?
    # For efficiency, we'll use a different approach: 
    # For each codeword, compute the set of words it covers (Hamming ball of radius 1).
    # Then for removal, check if removing a codeword leaves any word uncovered.
    
    # Precompute: for each codeword, what words does it cover?
    # A word w is covered by codeword c if Hamming distance <= radius.
    # For radius=1, w is covered by c if w==c or w differs in exactly 1 position.
    
    def get_covered_words(cw, q, n, radius):
        """Return a frozenset of tuples of words covered by codeword cw."""
        covered = set()
        covered.add(tuple(cw))
        if radius >= 1:
            for i in range(n):
                for v in range(q):
                    if v != cw[i]:
                        w = list(cw)
                        w[i] = v
                        covered.add(tuple(w))
        return frozenset(covered)
    
    # For large spaces, we can't store all covered words explicitly.
    # Instead, use a counting approach: count how many codewords cover each word.
    # Space size is q^n = 3^6 = 729, which is manageable.
    
    # Count coverage: for each word in the space, how many codewords are within radius?
    space_size = q ** n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx
    
    def idx_to_word(idx, q, n):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    # Build coverage count array
    coverage_count = [0] * space_size
    
    # For each codeword, increment coverage for all words within radius
    for cw in codewords:
        cw_idx = word_to_idx(cw)
        coverage_count[cw_idx] += 1
        if radius >= 1:
            for i in range(n):
                for v in range(q):
                    if v != cw[i]:
                        w = list(cw)
                        w[i] = v
                        coverage_count[word_to_idx(w)] += 1
    
    # Greedy removal: try to remove codewords that don't leave any word uncovered
    # A codeword can be removed if for every word it uniquely covers, there's another codeword covering it.
    # i.e., removing it doesn't drop any coverage_count to 0.
    
    removed = [False] * len(codewords)
    
    # Sort codewords by "redundancy" - those covering more already-well-covered words first
    # Simple heuristic: try removing in random order, or by some measure
    
    # Try multiple passes
    max_passes = 20
    for _pass in range(max_passes):
        if time.time() > deadline - 5:
            break
        removed_any = False
        # Try each codeword (skip already removed)
        for ci in range(len(codewords)):
            if removed[ci]:
                continue
            cw = codewords[ci]
            # Check if removing this codeword would leave any word uncovered
            # i.e., is there a word with coverage_count == 1 that is covered by this codeword?
            cw_idx = word_to_idx(cw)
            if coverage_count[cw_idx] == 1:
                continue  # This codeword is the only one covering itself
            if radius >= 1:
                can_remove = True
                for i in range(n):
                    for v in range(q):
                        if v != cw[i]:
                            w = list(cw)
                            w[i] = v
                            if coverage_count[word_to_idx(w)] == 1:
                                can_remove = False
                                break
                    if not can_remove:
                        break
                if not can_remove:
                    continue
            # Remove this codeword
            removed[ci] = True
            removed_any = True
            # Update coverage counts
            coverage_count[cw_idx] -= 1
            if radius >= 1:
                for i in range(n):
                    for v in range(q):
                        if v != cw[i]:
                            w = list(cw)
                            w[i] = v
                            coverage_count[word_to_idx(w)] -= 1
        if not removed_any:
            break
    
    # Build result
    result = [cw for i, cw in enumerate(codewords) if not removed[i]]
    
    # Now try local search: swap codewords to reduce count further
    # Try replacing a codeword with a new one that covers more uncovered words
    # This is more complex; for now, try a simple approach:
    # Try adding a new codeword and then removing multiple old ones if possible
    
    # Actually, let's try a different approach: iterative improvement
    # Try to find a codeword that, when added, allows removal of 2+ existing codewords
    
    if time.time() < deadline - 10:
        # Try local search with swaps
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Try adding a random codeword and see if we can remove more than 1
            # Sample candidate codewords
            improved = False
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random candidate
                cand = [random.randint(0, q - 1) for _ in range(n)]
                if cand in result:
                    continue
                
                # Compute coverage if we add cand
                # New coverage count
                new_cov = list(coverage_count)
                cand_idx = word_to_idx(cand)
                new_cov[cand_idx] += 1
                if radius >= 1:
                    for i in range(n):
                        for v in range(q):
                            if v != cand[i]:
                                w = list(cand)
                                w[i] = v
                                new_cov[word_to_idx(w)] += 1
                
                # Now try to greedily remove codewords from result
                # Count how many we can remove
                removable = 0
                # Build a temporary removal set
                temp_removed = [False] * len(result)
                for ri, rcw in enumerate(result):
                    rcw_idx = word_to_idx(rcw)
                    if new_cov[rcw_idx] <= 1:
                        continue
                    can_remove = True
                    if radius >= 1:
                        for i in range(n):
                            for v in range(q):
                                if v != rcw[i]:
                                    w = list(rcw)
                                    w[i] = v
                                    if new_cov[word_to_idx(w)] <= 1:
                                        can_remove = False
                                        break
                            if not can_remove:
                                break
                    if can_remove:
                        temp_removed[ri] = True
                        removable += 1
                        new_cov[rcw_idx] -= 1
                        if radius >= 1:
                            for i in range(n):
                                for v in range(q):
                                    if v != rcw[i]:
                                        w = list(rcw)
                                        w[i] = v
                                        new_cov[word_to_idx(w)] -= 1
                
                # We added 1, removed `removable`. Net change: 1 - removable
                # Accept if removable >= 2 (net decrease)
                if removable >= 2:
                    # Apply the change
                    result = [rcw for ri, rcw in enumerate(result) if not temp_removed[ri]]
                    result.append(cand)
                    coverage_count = new_cov
                    improved = True
                    break
            
            if not improved:
                break
    
    return result
# EVOLVE-BLOCK-END
