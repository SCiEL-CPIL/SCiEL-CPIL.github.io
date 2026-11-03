import random
import numpy as np
import time
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Optimized for q=3, n=8, radius=1.
    """
    if n == 0:
        return [[]]
    
    # Specific optimization for the known task instance q=3, n=8, radius=1
    # General fallback for other parameters
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors
    # For q=3, n=8, radius=1, each word has 1 + 8*2 = 17 neighbors.
    neighbors = [None] * total_words
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.append(word_to_int(new_word, q))
        neighbors[w] = neigh
    
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    # Greedy Construction
    def fast_greedy():
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        used_words = set()
        
        # Randomize the order of candidates to break symmetry
        # We pick the word that covers the most uncovered words
        # To speed up, we can sample or use a heap, but for 6561 words, 
        # a linear scan with vectorized count is fast enough.
        
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            
            # Random start index to break symmetry
            start_idx = random.randint(0, total_words - 1)
            
            # Linear scan
            for i in range(total_words):
                w = (start_idx + i) % total_words
                if w in used_words:
                    continue
                
                neigh_arr = neighbors_np[w]
                new_cov = np.sum(~covered[neigh_arr])
                
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
                    # Early termination if we cover all remaining
                    if new_cov == uncovered_count:
                        break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            used_words.add(best_word)
            covered[neighbors_np[best_word]] = True
            uncovered_count -= best_new_coverage
        
        return codewords

    def optimize(codewords, time_limit):
        if not codewords:
            return codewords
            
        # Initialize coverage counts
        coverage_count = np.zeros(total_words, dtype=np.int32)
        cw_set = set(codewords)
        
        for cw in codewords:
            coverage_count[neighbors_np[cw]] += 1
            
        # Helper to try removing redundant codewords
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                # Shuffle to avoid bias
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    neigh_arr = neighbors_np[cw]
                    # If any neighbor is covered by exactly 1 codeword (this one), it's not redundant
                    if np.any(coverage_count[neigh_arr] == 1):
                        continue
                    # Safe to remove
                    coverage_count[neigh_arr] -= 1
                    codewords.pop(i)
                    cw_set.discard(cw)
                    changed = True
                    removed_any = True
                    break
            return removed_any

        # Initial redundancy removal
        try_remove_redundant()
        
        # Local Search
        # We try to swap codewords to create redundancy, then remove.
        # Or we can use a "kick" strategy.
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
                
            # Try swaps
            # Pick a random codeword
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words uniquely covered by old_cw
            old_neigh = neighbors_np[old_cw]
            unique_mask = coverage_count[old_neigh] == 1
            unique_words = old_neigh[unique_mask]
            
            if len(unique_words) == 0:
                # Should have been removed, but just in case
                coverage_count[old_neigh] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Find a replacement
            best_new_cw = -1
            # We want a new_cw that covers all unique_words.
            # To make it fast, we can check candidates.
            # Since the space is small, we can iterate through all, but it's slow.
            # Better: Pick a random word from unique_words, and only consider candidates covering it.
            # But we don't have inverse neighbors precomputed.
            # Let's just sample candidates or iterate if time permits.
            # Given 6561 words, iterating all is ~6561 checks.
            # Each check involves np.isin or similar.
            
            # Optimization: 
            # If unique_words is small, we can intersect the neighbor sets of unique_words?
            # No, we need a word whose neighbor set is a superset of unique_words.
            
            # Let's try a few random candidates first, then full scan if needed?
            # Or just full scan with early break if we find a "good" one.
            # "Good" means it covers unique_words and minimizes new unique words.
            
            # To speed up:
            # 1. Filter candidates that cover the first unique word.
            # 2. Check if they cover the rest.
            
            u0 = unique_words[0]
            # Candidates must be in neighbors[u0]
            # Wait, neighbors[u0] are words within distance 1 of u0.
            # If new_cw covers u0, then new_cw is in neighbors[u0].
            # So we only need to check words in neighbors[u0].
            # Size of neighbors[u0] is 17.
            # This is MUCH faster than scanning all 6561.
            
            candidates = neighbors[u0]
            
            for new_cw in candidates:
                if new_cw in cw_set:
                    continue
                if new_cw == old_cw:
                    continue
                
                new_neigh = neighbors_np[new_cw]
                
                # Check if new_cw covers all unique_words
                # unique_words is a numpy array
                # We can check if set(unique_words) is subset of set(new_neigh)
                # Or use np.isin
                if not np.all(np.isin(unique_words, new_neigh)):
                    continue
                
                # Calculate cost: number of words that will become uniquely covered by new_cw
                # After swap:
                # - Words in neighbors[old_cw] lose 1 coverage
                # - Words in neighbors[new_cw] gain 1 coverage
                
                # We care about words in new_neigh.
                # Words in both old and new keep same count.
                # Words only in new get +1.
                
                in_both = np.isin(new_neigh, old_neigh)
                # Future counts for words in new_neigh
                # If in both: count stays same (was count, -1+1)
                # If only in new: count increases by 1
                # We want to avoid creating new unique words (count == 1)
                
                # Current count is coverage_count
                # For words only in new, future count = coverage_count[w] + 1
                # For words in both, future count = coverage_count[w]
                
                # We want to minimize the number of words in new_neigh that have future count == 1
                # AND are not in old_neigh? 
                # Actually, if a word is in both, its count doesn't change.
                # If it was 1, it stays 1.
                # If it was 2, it stays 2.
                # So words in both don't create NEW unique words, but they might be unique already.
                
                # Words only in new:
                # If current count is 0, becomes 1 (NEW unique)
                # If current count is 1, becomes 2 (not unique)
                
                # So we want to minimize the number of words in new_neigh \ old_neigh 
                # that have current count 0.
                
                mask_only_new = ~in_both
                # Count of words in new_neigh not in old_neigh with count 0
                cost = np.sum((coverage_count[new_neigh] == 0) & mask_only_new)
                
                # Also, we should prefer swaps that don't increase the total number of unique words?
                # The cost above is a good proxy.
                
                if best_new_cw == -1 or cost == 0:
                    best_new_cw = new_cw
                    if cost == 0:
                        break
            
            if best_new_cw != -1:
                # Perform swap
                coverage_count[old_neigh] -= 1
                new_neigh = neighbors_np[best_new_cw]
                coverage_count[new_neigh] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
            else:
                # No good swap found. Perturb.
                # Remove a random codeword and fill the gap.
                if random.random() < 0.1: # 10% chance to perturb if stuck
                    # Pick a random codeword to remove
                    rem_idx = random.randint(0, len(codewords) - 1)
                    rem_cw = codewords[rem_idx]
                    coverage_count[neighbors_np[rem_cw]] -= 1
                    codewords.pop(rem_idx)
                    cw_set.discard(rem_cw)
                    
                    # Find uncovered words
                    uncovered_mask = coverage_count == 0
                    uncovered_words = np.where(uncovered_mask)[0]
                    
                    if len(uncovered_words) > 0:
                        # Find best word to cover most uncovered
                        best_fill = -1
                        best_cov = -1
                        # Sample some candidates or scan all?
                        # Scan all is 6561 * 17 ops. Fast enough.
                        for w in range(total_words):
                            if w in cw_set:
                                continue
                            neigh_arr = neighbors_np[w]
                            cov = np.sum(uncovered_mask[neigh_arr])
                            if cov > best_cov:
                                best_cov = cov
                                best_fill = w
                                if best_cov == len(uncovered_words):
                                    break
                        
                        if best_fill != -1:
                            codewords.append(best_fill)
                            cw_set.add(best_fill)
                            coverage_count[neighbors_np[best_fill]] += 1
                            try_remove_redundant()
        
        try_remove_redundant()
        return codewords

    best_codewords = []
    
    # Run multiple restarts if time permits
    # Each greedy + optimize cycle should take a few seconds.
    
    while time.time() - start_time < time_limit - 5:
        cw = fast_greedy()
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            cw = optimize(cw, remaining_time - 5)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
            # If we hit the lower bound, we can stop? 
            # Lower bound is 403.
            if len(cw) <= 403:
                break

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
