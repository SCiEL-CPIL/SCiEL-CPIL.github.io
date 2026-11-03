"""Improved covering code construction using bitmasks for efficient coverage tracking."""
import random
import time
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses bitmask-based greedy + local search with multiple restarts.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    if total_words == 0:
        return []
    
    # For small spaces, we can use numpy arrays for coverage
    # For the specific case q=3, n=8, radius=1: total_words=6561
    # Each codeword covers 1 + n*(q-1) = 1 + 8*2 = 17 words
    
    # Precompute: for each word index, its neighbors (within Hamming distance <= radius)
    # Represent words as integers 0..total_words-1
    # A word w corresponds to digits: w % q, (w//q) % q, ...
    
    def int_to_digits(w):
        """Convert integer to list of digits in base q, length n."""
        digits = [0] * n
        for i in range(n):
            digits[i] = w % q
            w //= q
        return digits
    
    def digits_to_int(digits):
        """Convert list of digits to integer in base q."""
        val = 0
        for i in range(n - 1, -1, -1):
            val = val * q + digits[i]
        return val
    
    # Precompute neighbors for all words
    # neighbors[w] = list of word indices within Hamming distance <= radius from w
    neighbors = []
    for w in range(total_words):
        digits = int_to_digits(w)
        neigh_set = set()
        neigh_set.add(w)
        
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    digits[pos] = new_val
                    neigh_set.add(digits_to_int(digits))
                digits[pos] = orig
        
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(n):
                    if pos1 >= pos2:
                        continue
                    orig1 = digits[pos1]
                    orig2 = digits[pos2]
                    for v1 in range(q):
                        if v1 == orig1:
                            continue
                        for v2 in range(q):
                            if v2 == orig2:
                                continue
                            digits[pos1] = v1
                            digits[pos2] = v2
                            neigh_set.add(digits_to_int(digits))
                    digits[pos1] = orig1
                    digits[pos2] = orig2
        
        neighbors.append(list(neigh_set))
    
    # Precompute neighbor sets as frozensets for fast membership
    neighbor_sets = [frozenset(neigh) for neigh in neighbors]
    
    # Also precompute: for each word, which codeword positions cover it
    # We'll track coverage count per word
    
    def greedy_construct():
        """Greedy set cover: repeatedly pick the word covering most uncovered words."""
        covered = np.zeros(total_words, dtype=np.int8)
        codewords = []
        uncovered = total_words
        
        while uncovered > 0:
            best_w = -1
            best_gain = -1
            
            # Sample candidates for speed
            if uncovered > total_words * 0.05:
                sample_size = min(2000, total_words)
                candidates = random.sample(range(total_words), sample_size)
            else:
                candidates = list(range(total_words))
            
            for w in candidates:
                gain = 0
                for u in neighbors[w]:
                    if covered[u] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_w = w
            
            if best_w == -1 or best_gain == 0:
                # Fallback: pick any uncovered word
                for w in range(total_words):
                    if covered[w] == 0:
                        best_w = w
                        break
                if best_w == -1:
                    break
            
            codewords.append(best_w)
            for u in neighbors[best_w]:
                if covered[u] == 0:
                    covered[u] = 1
                    uncovered -= 1
        
        return codewords
    
    def try_remove_redundant(codewords, coverage):
        """Remove any codeword that doesn't uniquely cover any word."""
        removed = True
        while removed:
            removed = False
            i = 0
            while i < len(codewords):
                cw = codewords[i]
                # Check if any word is covered only by this codeword
                unique = False
                for u in neighbors[cw]:
                    if coverage[u] == 1:
                        unique = True
                        break
                if not unique:
                    # Remove it
                    for u in neighbors[cw]:
                        coverage[u] -= 1
                    codewords.pop(i)
                    removed = True
                    # Do not increment i, because the next element shifted into position i
                else:
                    i += 1
        return codewords
    
    def local_search(codewords, time_limit):
        """Local search: try to remove redundant codewords, then swap to reduce count."""
        coverage = np.zeros(total_words, dtype=np.int16)
        for cw in codewords:
            for u in neighbors[cw]:
                coverage[u] += 1
        
        codewords = try_remove_redundant(codewords, coverage)
        
        start_time = time.time()
        no_improve = 0
        best_len = len(codewords)
        
        while time.time() - start_time < time_limit:
            # Try to remove redundant
            before = len(codewords)
            codewords = try_remove_redundant(codewords, coverage)
            if len(codewords) < before:
                no_improve = 0
                if len(codewords) < best_len:
                    best_len = len(codewords)
                continue
            
            if len(codewords) == 0:
                break
            
            # Try swaps: remove one codeword, add another that covers all previously-uncovered words
            found_improvement = False
            for _ in range(50):  # Try multiple swap attempts per iteration
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                
                # Find words that would become uncovered if we remove old_cw
                would_uncover = []
                for u in neighbors[old_cw]:
                    if coverage[u] == 1:
                        would_uncover.append(u)
                
                if not would_uncover:
                    # Can remove directly
                    for u in neighbors[old_cw]:
                        coverage[u] -= 1
                    codewords.pop(idx)
                    found_improvement = True
                    break
                
                # Need to find a new codeword that covers all would_uncover
                # Candidate: neighbors of a random word in would_uncover
                target = random.choice(would_uncover)
                candidates = list(neighbors[target])
                random.shuffle(candidates)
                
                best_new_cw = -1
                best_score = -1
                
                for new_cw in candidates:
                    if new_cw == old_cw:
                        continue
                    # Check if new_cw covers all would_uncover
                    valid = True
                    for u in would_uncover:
                        if u not in neighbor_sets[new_cw]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: minimize new coverage overlap (prefer codeword that adds least redundancy)
                    # Or maximize: how many words it covers that are already well-covered
                    score = 0
                    for u in neighbors[new_cw]:
                        if u not in neighbor_sets[old_cw]:
                            score += coverage[u]
                    
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    # Perform swap
                    for u in neighbors[old_cw]:
                        coverage[u] -= 1
                    for u in neighbors[best_new_cw]:
                        coverage[u] += 1
                    codewords[idx] = best_new_cw
                    found_improvement = True
                    break
            
            if found_improvement:
                no_improve = 0
                if len(codewords) < best_len:
                    best_len = len(codewords)
            else:
                no_improve += 1
                if no_improve > 100:
                    break
        
        return codewords
    
    # Main: multiple restarts
    start_time = time.time()
    total_time_limit = 110  # Leave 40s safety margin for 150s total
    best_codewords = None
    best_len = float('inf')
    
    while time.time() - start_time < total_time_limit:
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Allocate time for this restart
        opt_time = min(remaining - 2, 30)
        if opt_time < 2:
            break
        
        # Greedy initialization
        cw = greedy_construct()
        
        # Local search optimization
        cw = local_search(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw[:]
        
        # If we hit a good solution, do one more polish
        if best_len <= 490:
            # Try to polish the best solution
            polish_time = min(remaining - 3, 15)
            if polish_time > 2:
                cw_polished = local_search(best_codewords[:], polish_time)
                if len(cw_polished) < best_len:
                    best_len = len(cw_polished)
                    best_codewords = cw_polished[:]
        
        if time.time() - start_time > total_time_limit - 5:
            break
    
    if best_codewords is None:
        best_codewords = greedy_construct()
    
    # Convert to list of digit lists
    result = [int_to_digits(cw) for cw in best_codewords]
    return result
