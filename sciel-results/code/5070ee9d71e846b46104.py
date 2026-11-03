import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words
        all_words = np.array(list(itertools.product(range(q), repeat=n)))
        
        # Precompute coverage matrix: coverage[i][j] is True if dist(i,j) <= radius
        # dist <= 1 means i==j or they differ in 1 position
        # This is a 729x729 boolean matrix
        # diff = np.abs(all_words[:, None, :] - all_words[None, :, :])
        # dist = (diff != 0).sum(axis=2)
        # coverage = dist <= radius
        
        # Faster way:
        # A word covers itself and 6*(q-1) neighbors
        coverage = np.zeros((total_words, total_words), dtype=bool)
        for i in range(total_words):
            w = all_words[i]
            coverage[i, i] = True
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = w.copy()
                    new_w[pos] = val
                    # Find index of new_w
                    # Since we generated in lexicographic order, index is computable
                    # idx = val * q^(n-1-pos) + ...
                    # Or just use a dictionary
                    pass
        
        # Dictionary for index lookup is faster for building coverage
        index_map = {tuple(w): i for i, w in enumerate(all_words)}
        
        for i in range(total_words):
            w = all_words[i]
            coverage[i, i] = True
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = w.copy()
                    new_w[pos] = val
                    j = index_map[tuple(new_w)]
                    coverage[i, j] = True

        def greedy_init():
            uncovered = set(range(total_words))
            code = []
            # Randomize order for diversity? No, greedy is deterministic usually.
            # We will run multiple greedy inits with different random seeds/orderings
            return code, uncovered
            
        def run_greedy(seed):
            rng = random.Random(seed)
            uncovered = set(range(total_words))
            code = []
            # Precompute coverage counts for all words?
            # count[i] = number of uncovered words covered by i
            counts = np.array([len(coverage[i]) for i in range(total_words)], dtype=int)
            
            while uncovered:
                # Find word with max counts
                # To add randomness, pick from top K
                top_k = 10
                # Get indices of top k
                # This is O(N) to find max, or use partial sort
                # N=729, so O(N) is fine
                best_idx = -1
                best_val = -1
                candidates = []
                
                for i in range(total_words):
                    if counts[i] > best_val:
                        best_val = counts[i]
                        best_idx = i
                        candidates = [i]
                    elif counts[i] == best_val and counts[i] > 0:
                        candidates.append(i)
                
                if best_idx == -1: break
                
                chosen = rng.choice(candidates)
                code.append(chosen)
                uncovered -= set(np.where(coverage[chosen])[0])
                
                # Update counts for remaining uncovered
                # For each i, counts[i] -= (coverage[i] & coverage[chosen]).sum()
                # This is O(N^2) per step. 729^2 * 80 steps ~ 40M ops. Might be slow in Python loops.
                # Use numpy
                chosen_cov = coverage[chosen]
                # counts = counts - (coverage[:, chosen_cov] & chosen_cov[None, :]).sum(axis=1) ??
                # coverage is 729x729 bool
                # We want for each row i, the sum of coverage[i] & coverage[chosen]
                # This is the dot product of row i of coverage and column chosen of coverage? No.
                # It is the number of common neighbors.
                # counts[i] = sum(coverage[i, j] for j in uncovered)
                # When we remove coverage[chosen] from uncovered, 
                # new_counts[i] = old_counts[i] - sum(coverage[i, j] for j in coverage[chosen] if j was in uncovered)
                # Since coverage[chosen] is a subset of uncovered (mostly), 
                # new_counts[i] = old_counts[i] - (coverage[i] & coverage[chosen]).sum()
                
                # Vectorized update:
                # intersection = coverage & coverage[chosen][None, :]
                # reduction = intersection.sum(axis=1)
                # counts -= reduction
                # But we only care about words still in uncovered? No, counts are for all words.
                
                reduction = (coverage & coverage[chosen][None, :]).sum(axis=1)
                counts -= reduction
                
            return code

        def local_search(code):
            code_set = set(code)
            # Initialize counts
            counts = np.zeros(total_words, dtype=int)
            for c in code_set:
                counts[coverage[c]] += 1
            
            # Check validity
            # min(counts) should be >= 1
            
            improved = True
            while improved:
                improved = False
                # 1. Try to remove
                for c in list(code_set):
                    if np.min(counts[coverage[c]]) > 1:
                        code_set.remove(c)
                        counts[coverage[c]] -= 1
                        improved = True
                        break # Restart loop after change
                
                if improved: continue
                
                # 2. Try to swap
                # For each c in code, find w not in code such that swapping reduces size or maintains size but improves "slack"?
                # We want to minimize size. So swap is only useful if we can remove 2 and add 1? No, swap is 1 for 1.
                # Swap is useful if it allows a subsequent removal.
                # Or if we are stuck at size S, and we find a swap that leads to a state where a removal is possible.
                
                # Heuristic: Try swapping c with w if w covers all "private" words of c.
                current_list = list(code_set)
                random.shuffle(current_list)
                
                swapped = False
                for c in current_list:
                    private = [w for w in np.where(coverage[c])[0] if counts[w] == 1]
                    if not private:
                        # Can remove, handled above
                        continue
                    
                    # Find w not in code_set that covers all private
                    # Optimize: Only check w that are neighbors of private words?
                    # private words are covered by c. w must cover them.
                    # So w must be in intersection of coverage[private[0]], coverage[private[1]], ...
                    # And w must not be in code_set.
                    
                    # Start with all words, intersect with coverage of each private word
                    candidates = set(range(total_words))
                    for p in private:
                        candidates &= set(np.where(coverage[p])[0])
                    
                    # Remove existing code
                    candidates -= code_set
                    
                    if candidates:
                        # Pick one, maybe the one with highest coverage count?
                        best_w = max(candidates, key=lambda x: counts[x]) # Heuristic
                        # Perform swap
                        code_set.remove(c)
                        counts[coverage[c]] -= 1
                        code_set.add(best_w)
                        counts[coverage[best_w]] += 1
                        swapped = True
                        improved = True
                        break
                        
                if not swapped:
                    break
                    
            return code_set

        best_code = None
        best_size = float('inf')
        start_time = time.time()
        time_limit = 110
        
        # Multiple restarts
        for seed in range(100):
            if time.time() - start_time > time_limit: break
            init_code = run_greedy(seed)
            ls_code = local_search(init_code)
            if len(ls_code) < best_size:
                best_size = len(ls_code)
                best_code = ls_code
                
        return [list(all_words[i]) for i in best_code]
    else:
        # Fallback
        pass
