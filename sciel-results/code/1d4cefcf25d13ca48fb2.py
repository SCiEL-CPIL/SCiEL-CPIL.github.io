"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts.
    """
    if n == 0:
        return [[]]
    
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
    
    all_words = list(range(total_words))
    
    # Precompute neighbors as lists for faster iteration
    neighbors = [None] * total_words
    for w in all_words:
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
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            new_int = word_to_int(new_word, q)
                            if new_int not in neigh:
                                neigh.append(new_int)
        neighbors[w] = neigh
    
    # Convert to sets for fast membership testing
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # Precompute neighbor list as numpy arrays for faster operations
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    def fast_greedy():
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        used_words = set()
        
        while uncovered_count > 0:
            best_word = -1
            best_new_coverage = -1
            
            # Randomize start to break symmetry
            start_idx = random.randint(0, total_words - 1)
            
            for i in range(total_words):
                w = (start_idx + i) % total_words
                if w in used_words:
                    continue
                
                # Vectorized coverage count
                neigh_arr = neighbors_np[w]
                new_cov = np.sum(~covered[neigh_arr])
                
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = w
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
        # Initialize coverage counts
        coverage_count = np.zeros(total_words, dtype=np.int32)
        cw_set = set(codewords)
        
        for cw in codewords:
            coverage_count[neighbors_np[cw]] += 1
                
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    # Check if any word is uniquely covered by this codeword
                    neigh_arr = neighbors_np[cw]
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

        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
                
            # Pick a random codeword to potentially swap
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            
            # Find words uniquely covered by old_cw
            old_neigh = neighbors_np[old_cw]
            unique_mask = coverage_count[old_neigh] == 1
            unique_words = old_neigh[unique_mask]
            
            if len(unique_words) == 0:
                # Can remove
                coverage_count[old_neigh] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Try to find a better replacement
            # The replacement must cover all unique_words
            best_new_cw = -1
            best_cost = float('inf')
            
            # Sample a subset of candidates for speed
            # First try all, but with early termination
            for new_cw in all_words:
                if new_cw in cw_set or new_cw == old_cw:
                    continue
                
                new_neigh = neighbors_np[new_cw]
                
                # Check if new_cw covers all unique_words
                if not np.all(np.isin(unique_words, new_neigh)):
                    continue
                
                # Calculate cost: number of words that will be uniquely covered by new_cw
                # After swap:
                # - Words in neighbors[old_cw] lose 1 coverage
                # - Words in neighbors[new_cw] gain 1 coverage
                
                # Compute future coverage for words in new_neigh
                # Words that are in both old and new keep same count
                # Words only in new get +1
                in_both = np.isin(new_neigh, old_neigh)
                future_counts = np.where(in_both, coverage_count[new_neigh], coverage_count[new_neigh] + 1)
                
                cost = np.sum(future_counts == 1)
                
                if cost < best_cost:
                    best_cost = cost
                    best_new_cw = new_cw
                    
                    # If cost is 0, it's a perfect swap (no new unique words)
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
                # No valid swap found for this codeword. 
                # Skip this move.
                pass
        
        try_remove_redundant()
        return codewords

    # Main loop
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    # Run one greedy and optimize
    cw = fast_greedy()
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        cw = optimize(cw, remaining_time - 5)
    
    best_codewords = cw
    
    # If time remains, try another restart?
    # Given the cost of greedy, maybe just one restart is safer to avoid timeout.
    # But the local search is the main driver.
    # Let's try to run another greedy if we have plenty of time left.
    
    while time.time() - start_time < time_limit - 10:
        cw = fast_greedy()
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            cw = optimize(cw, remaining_time - 5)
        
        if len(cw) < len(best_codewords):
            best_codewords = cw
            
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
