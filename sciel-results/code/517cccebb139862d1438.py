import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as numpy arrays for fast operations
    # Each neighbor list has size 1 + 8*2 = 17
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Helper: Compute coverage counts
    def compute_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    # Helper: Reduce code by removing redundant codewords
    def reduce_code(code, counts):
        if not code:
            return list(code), counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_C = []
    best_size = float('inf')
    best_counts = None

    # Run multiple randomized greedy initializations followed by local search
    # Since the problem is symmetric, we don't need many restarts if the local search is strong,
    # but 2-3 diverse starts help.
    
    num_restarts = 3
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit - 20: # Reserve time for final check
            break
            
        # 1. Randomized Greedy Initialization
        uncovered = np.ones(N, dtype=bool)
        C = []
        
        while np.any(uncovered):
            # Find max coverage
            best_count = -1
            candidates = []
            # Optimization: We can vectorize the sum if we keep neighbors in a 2D array, 
            # but 6561 * 17 is small. The bottleneck is the Python loop over 6561 items.
            # Let's try to speed up the scan.
            # Actually, scanning 6561 items with a numpy sum on a 17-element array is quite fast in PyPy/NumPy.
            # But we can do better by maintaining a list of active candidates? 
            # For simplicity and robustness, we stick to the scan but randomize ties.
            
            for i in range(N):
                if not uncovered[i]:
                    # If the word itself is covered, it doesn't help cover *uncovered* points 
                    # unless it covers other uncovered points. 
                    # Wait, the greedy step is: pick c that maximizes sum(uncovered[neighbors[c]]).
                    # If uncovered[i] is False, it doesn't mean neighbors[i] are all covered.
                    # So we must check all i.
                    pass
                
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    candidates = [i]
                elif cnt == best_count:
                    candidates.append(i)
            
            if best_count == 0:
                break
                
            # Randomly pick from candidates to diversify
            best_idx = random.choice(candidates)
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
            if time.time() - start_time > time_limit - 10:
                break
        
        # Initial Reduction
        counts = compute_counts(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()

        # 2. Local Search
        # Strategy:
        # Loop:
        #   a. Reduce
        #   b. If reduced, continue
        #   c. If not reduced, try to find a swap (1-for-1) that allows reduction
        #   d. If no improving swap, try a neutral swap (1-for-1) that doesn't increase size 
        #      but changes the set, hoping to escape local minimum.
        #   e. If no swap, random kick (replace a random codeword with a random neighbor)
        
        current_C = C
        current_counts = counts
        stuck_counter = 0
        
        while time.time() - start_time < time_limit - 5:
            old_size = len(current_C)
            current_C, current_counts = reduce_code(current_C, current_counts)
            
            if len(current_C) < best_size:
                best_size = len(current_C)
                best_C = current_C.copy()
                best_counts = current_counts.copy()
                stuck_counter = 0 # Reset on improvement
            
            if len(current_C) == old_size:
                # No reduction occurred. Try swaps.
                improved_this_iter = False
                indices = list(range(len(current_C)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() - start_time > time_limit - 5:
                        break
                    
                    c = current_C[idx]
                    u_indices = neighbors[c]
                    unique_mask = current_counts[u_indices] == 1
                    unique_points = u_indices[unique_mask]
                    
                    if len(unique_points) == 0:
                        # Should have been removed, force it
                        current_C.remove(c)
                        current_counts[neighbors[c]] -= 1
                        improved_this_iter = True
                        break
                    
                    # If too many unique points, hard to cover with one swap
                    if len(unique_points) > 3:
                        continue
                    
                    # Find candidates c' that cover all unique_points
                    # Intersection of neighbors[unique_point]
                    if len(unique_points) == 1:
                        candidates_set = set(neighbors[unique_points[0]].tolist())
                    else:
                        candidates_set = set(neighbors[unique_points[0]].tolist())
                        for u in unique_points[1:]:
                            candidates_set &= set(neighbors[u].tolist())
                            if not candidates_set:
                                break
                    
                    if not candidates_set:
                        continue
                    
                    # Candidates must not be in current_C
                    candidates_set.difference_update(current_C)
                    
                    if not candidates_set:
                        continue
                    
                    # Heuristic: Pick candidate that maximizes overlap with other codewords?
                    # Or just pick randomly. Let's pick randomly to explore.
                    c_prime = random.choice(list(candidates_set))
                    
                    # Apply swap
                    old_C = current_C.copy()
                    old_counts = current_counts.copy()
                    
                    current_C.remove(c)
                    current_C.append(c_prime)
                    
                    current_counts[neighbors[c]] -= 1
                    current_counts[neighbors[c_prime]] += 1
                    
                    if np.any(current_counts < 1):
                        # Invalid swap, revert
                        current_C = old_C
                        current_counts = old_counts
                        continue
                    
                    # Reduce after swap
                    new_C, new_counts = reduce_code(current_C, current_counts)
                    
                    if len(new_C) < len(old_C):
                        # Improvement!
                        current_C = new_C
                        current_counts = new_counts
                        improved_this_iter = True
                        break
                    else:
                        # Neutral swap: keep it to change landscape
                        # This is key to escaping local minima
                        current_C = new_C
                        current_counts = new_counts
                        improved_this_iter = True
                        break
                
                if not improved_this_iter:
                    # No valid swap found. Try a random kick.
                    # Replace a random codeword with a random neighbor.
                    if current_C:
                        c_idx = random.randint(0, len(current_C) - 1)
                        c = current_C[c_idx]
                        cand = neighbors[c]
                        c_prime = cand[random.randint(0, len(cand) - 1)]
                        
                        if c_prime not in current_C:
                            old_C = current_C.copy()
                            old_counts = current_counts.copy()
                            
                            current_C.remove(c)
                            current_C.append(c_prime)
                            
                            current_counts[neighbors[c]] -= 1
                            current_counts[neighbors[c_prime]] += 1
                            
                            if not np.any(current_counts < 1):
                                # Accept kick, then reduce
                                current_C, current_counts = reduce_code(current_C, current_counts)
                            else:
                                current_C = old_C
                                current_counts = old_counts
                    else:
                        break
            else:
                # Reduction happened, reset stuck counter logic implicitly
                pass

    # Final verification and return
    if not best_C:
        # Fallback if something went wrong
        return [list(all_words[i]) for i in C]

    # Ensure distinctness (should be guaranteed by set logic, but just in case)
    unique_best_C = list(set(best_C))
    
    return [list(all_words[i]) for i in unique_best_C]
