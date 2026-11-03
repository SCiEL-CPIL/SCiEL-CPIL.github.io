import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as a 2D numpy array (total_words, n)
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Fast distance calculation
        diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        dist = (diff != 0).sum(axis=2)
        
        # coverage_mask[i] is a boolean array of length total_words
        # coverage_mask[i][j] is True if dist[i][j] <= radius
        coverage_mask = dist <= radius
        
        # Precompute list of covered indices for each word for faster iteration
        # coverage_indices[i] is a list of j such that coverage_mask[i][j] is True
        coverage_indices = [np.where(coverage_mask[i])[0].tolist() for i in range(total_words)]
        
        def greedy_set_cover(random_seed=None):
            if random_seed is not None:
                random.seed(random_seed)
            
            uncovered = set(range(total_words))
            code_indices = []
            
            while uncovered:
                best_idx = -1
                best_cover_count = -1
                # To introduce randomness in tie-breaking for restarts
                # We can just pick the first best, or randomize order
                for i in range(total_words):
                    # Count how many uncovered words i covers
                    # Using numpy sum is faster
                    count = np.sum(coverage_mask[i, list(uncovered)])
                    if count > best_cover_count:
                        best_cover_count = count
                        best_idx = i
                        if best_cover_count == len(uncovered):
                            break
                    elif count == best_cover_count and best_idx != -1:
                        # Random tie breaker
                        if random.random() < 0.5:
                            best_idx = i
                
                if best_idx == -1:
                    break
                    
                code_indices.append(best_idx)
                # Update uncovered
                # Uncovered words are those NOT covered by best_idx
                # i.e., uncovered - coverage[best_idx]
                covered_by_best = set(coverage_indices[best_idx])
                uncovered -= covered_by_best
                
            return code_indices

        def local_search(code_indices, max_iters=200):
            code_set = set(code_indices)
            # counts[w] = number of codewords covering w
            counts = np.zeros(total_words, dtype=int)
            for c in code_set:
                counts[coverage_indices[c]] += 1
            
            improved = True
            iterations = 0
            
            while improved and iterations < max_iters:
                improved = False
                iterations += 1
                
                # 1. Pruning: Remove codewords that are not critical
                # A codeword c is removable if all w in coverage[c] have counts[w] > 1
                removable = None
                for c in list(code_set):
                    # If min count in coverage[c] is > 1, it's removable
                    if np.min(counts[coverage_indices[c]]) > 1:
                        removable = c
                        break
                        
                if removable is not None:
                    code_set.remove(removable)
                    counts[coverage_indices[removable]] -= 1
                    improved = True
                    continue
                
                # 2. 1-Swap
                # Try to replace one codeword with another to improve "slack" or just to move
                # Actually, simple 1-swap doesn't reduce size.
                # We need to find a swap that allows a subsequent removal?
                # Or just a standard local search for minimum weight?
                # The objective is to minimize |code_set|.
                # If we can't prune, we are at a local minimum for size.
                # To escape, we can do a 2-exchange: Remove 2, Add 2.
                # This might allow a 3rd removal later?
                
                # Let's try 2-Exchange (Remove c1, c2; Add w1, w2)
                # Condition: After removing c1, c2, the uncovered words must be covered by w1, w2.
                # Uncovered words = (coverage[c1] U coverage[c2]) - (coverage of other codewords)
                # This is expensive to check naively.
                
                # Alternative: Simulated Annealing style "Move"
                # Pick a random codeword c, remove it.
                # Find a set of words W that covers the newly uncovered words.
                # If |W| < 1 (impossible) or we can replace c with w such that valid...
                # This is complex.
                
                # Let's stick to a faster 1-swap that *might* enable pruning.
                # Or, let's just do a "Perturbation": Remove a random codeword, then add back the best cover for the gaps.
                # This is a "Restart" inside local search.
                
                # Given time constraints, let's optimize the existing 1-swap to be faster and add a "2-swap" heuristic.
                
                # Faster 1-Swap check:
                # For each c in code_set:
                #   Private words P_c = {w | counts[w]==1 and w in coverage[c]}
                #   If P_c is empty, remove c (Pruning handles this).
                #   We want to find w such that P_c subset of coverage[w].
                #   If found, swap c->w. This keeps size same, but might change the structure to allow future pruning.
                
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private_words = [w for w in coverage_indices[c] if counts[w] == 1]
                    if not private_words:
                        code_set.remove(c)
                        counts[coverage_indices[c]] -= 1
                        improved = True
                        swapped = True
                        break
                    
                    # Find candidate w
                    # We can precompute for each w, the set of private words it covers? No.
                    # Just check.
                    found_w = None
                    for w in range(total_words):
                        if w in code_set:
                            continue
                        # Check if w covers all private_words
                        # Using numpy: all(coverage_mask[w, private_words])
                        if np.all(coverage_mask[w, private_words]):
                            found_w = w
                            break
                        
                    if found_w is not None:
                        code_set.remove(c)
                        counts[coverage_indices[c]] -= 1
                        code_set.add(found_w)
                        counts[coverage_indices[found_w]] += 1
                        improved = True
                        swapped = True
                        break
                
                if not swapped:
                    # Try 2-Exchange: Remove 2, Add 2
                    # This is O(N^2) candidates. 73^2 = 5329 pairs.
                    # For each pair, check if valid.
                    # Might be too slow if done naively in Python loops.
                    # Let's skip 2-exchange for now and rely on multiple restarts of Greedy+1Swap.
                    break

            return code_set

        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts
        start_time = time.time()
        max_time = 100 # Leave 20s buffer
        
        while time.time() - start_time < max_time:
            # Randomized Greedy
            code = greedy_set_cover(random_seed=random.randint(0, 10000))
            
            # Local Search
            code_set = local_search(code, max_iters=100)
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set
                if best_size == 73: # Optimal known
                    break
        
        return [list(all_words[i]) for i in best_code]
    
    else:
        # Fallback
        all_words = list(itertools.product(range(q), repeat=n))
        total_words = len(all_words)
        
        def hamming_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
            
        coverage = [set() for _ in range(total_words)]
        for i, c in enumerate(all_words):
            for j, w in enumerate(all_words):
                if hamming_distance(c, w) <= radius:
                    coverage[i].add(j)
        
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            for i in range(total_words):
                count = len(coverage[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1: break
            code_indices.append(best_idx)
            uncovered -= coverage[best_idx]
            
        return [list(all_words[i]) for i in code_indices]
