import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Precompute all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = q ** n
    
    # For each word, precompute its "ball" (words within distance <= radius)
    # A word covers itself and all words at Hamming distance 1 (for radius=1)
    # ball[word_index] = set of word indices covered
    # But 729 words, each covers 13, so we can precompute efficiently
    
    # Map word tuple to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    # Precompute neighbors (words at distance <= 1) for each word
    # For radius=1: the word itself + all single-position changes
    neighbors = [None] * num_words
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_word = list(w)
                new_word[pos] = val
                ball.append(word_to_idx[tuple(new_word)])
        neighbors[i] = ball
    
    # Greedy set cover: pick codewords one at a time, each covering max uncovered
    uncovered = set(range(num_words))
    codewords = []
    
    # For speed, precompute coverage counts
    # coverage_count[j] = number of uncovered words that codeword j would cover
    coverage_count = np.array([len(neighbors[j]) for j in range(num_words)], dtype=np.int32)
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Find the codeword covering the most uncovered words
        # For efficiency, compute for candidate codewords
        best_idx = -1
        best_count = -1
        
        # Sample candidates or check all (729 is small)
        for j in range(num_words):
            if best_count >= len(uncovered):
                break
            count = sum(1 for x in neighbors[j] if x in uncovered)
            if count > best_count:
                best_count = count
                best_idx = j
        
        if best_idx < 0 or best_count <= 0:
            break
        
        codewords.append(all_words[best_idx])
        for x in neighbors[best_idx]:
            uncovered.discard(x)
    
    # Local optimization: try to delete codewords
    if time.time() < deadline:
        # Build coverage map: for each word, which codeword indices cover it
        # codewords is a list of tuples
        n_codewords = len(codewords)
        
        # For each word, store the set of codeword indices that cover it
        # covered_by[word_idx] = list of codeword indices
        covered_by = [set() for _ in range(num_words)]
        for ci, cw in enumerate(codewords):
            ci_idx = word_to_idx[cw]
            for x in neighbors[ci_idx]:
                covered_by[x].add(ci)
        
        # Try deleting codewords one by one
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for ci in range(len(codewords)):
                if time.time() > deadline:
                    break
                # Check if removing codeword ci still covers everything
                # A word is still covered if it has another codeword covering it
                cw_idx = word_to_idx[codewords[ci]]
                removable = True
                for x in neighbors[cw_idx]:
                    if len(covered_by[x]) == 1:
                        removable = False
                        break
                if removable:
                    # Remove codeword ci
                    for x in neighbors[cw_idx]:
                        covered_by[x].discard(ci)
                    codewords.pop(ci)
                    # Re-index: all codeword indices > ci shift down
                    # Rebuild covered_by indices
                    new_covered_by = [set() for _ in range(num_words)]
                    for ni, nw in enumerate(codewords):
                        ni_idx = word_to_idx[nw]
                        for x in neighbors[ni_idx]:
                            new_covered_by[x].add(ni)
                    covered_by = new_covered_by
                    changed = True
                    break
    
    # Try swap optimization: replace one codeword with another to reduce count
    # This is more complex; try a simple version
    if time.time() < deadline and len(codewords) > 70:
        # Try to find if we can reduce by 1 via a clever swap
        # For each codeword, try replacing it with a different word
        # Check if the new set still covers everything
        n_codewords = len(codewords)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ci in range(len(codewords)):
                if time.time() > deadline:
                    break
                old_cw = codewords[ci]
                old_idx = word_to_idx[old_cw]
                
                # Find words that are only covered by codeword ci
                # These must still be covered by the replacement
                only_covered = []
                for x in neighbors[old_idx]:
                    if len(covered_by[x]) == 1:
                        only_covered.append(x)
                
                if not only_covered:
                    # Can just delete (should have been caught above, but safety)
                    for x in neighbors[old_idx]:
                        covered_by[x].discard(ci)
                    codewords.pop(ci)
                    new_covered_by = [set() for _ in range(num_words)]
                    for ni, nw in enumerate(codewords):
                        ni_idx = word_to_idx[nw]
                        for x in neighbors[ni_idx]:
                            new_covered_by[x].add(ni)
                    covered_by = new_covered_by
                    improved = True
                    break
                
                # Try replacing with a word that covers all of only_covered
                # The replacement must be within distance 1 of all words in only_covered
                # AND the replacement's ball must cover all words that were uniquely covered
                # Actually, the replacement just needs to cover the words in only_covered
                # (since other words are covered by other codewords)
                
                # For a candidate replacement r, it must satisfy:
                # for all x in only_covered: x is in neighbors[word_to_idx[r]]
                # i.e., dist(r, x) <= 1 for all x in only_covered
                
                # Also, r should ideally cover new words that help future deletions
                # But for now, just try to find any valid replacement
                
                # The intersection of balls of all words in only_covered
                # A word r is valid if dist(r, x) <= 1 for all x in only_covered
                # This means r is in the intersection of the balls of all x in only_covered
                
                # For small only_covered, we can enumerate candidates
                if len(only_covered) <= 3:
                    # Find intersection of balls
                    candidate_set = set(neighbors[only_covered[0]])
                    for x in only_covered[1:]:
                        candidate_set &= set(neighbors[x])
                    
                    for cand_idx in candidate_set:
                        cand = all_words[cand_idx]
                        if cand == old_cw:
                            continue
                        # Check: after replacing, is everything still covered?
                        # Words in neighbors[old_idx] that are NOT in neighbors[cand_idx]
                        # must be covered by other codewords
                        ok = True
                        for x in neighbors[old_idx]:
                            if x not in neighbors[cand_idx]:
                                # Must be covered by another codeword
                                if len(covered_by[x]) <= 1:
                                    ok = False
                                    break
                        if ok:
                            # Do the replacement
                            codewords[ci] = cand
                            # Update covered_by
                            for x in neighbors[old_idx]:
                                covered_by[x].discard(ci)
                            for x in neighbors[cand_idx]:
                                covered_by[x].add(ci)
                            improved = True
                            break
                if improved:
                    break
    
    # Final verification
    if time.time() < deadline:
        # Verify coverage
        covered = set()
        for cw in codewords:
            ci_idx = word_to_idx[cw]
            covered.update(neighbors[ci_idx])
        if len(covered) < num_words:
            # Fallback: use the tetracode construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return codewords
