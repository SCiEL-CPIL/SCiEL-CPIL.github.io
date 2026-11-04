import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        # Simple greedy for general case
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
    
    # Precompute neighbors
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
        
    # Convert neighbors to list of arrays for faster indexing
    # neighbors[i] is an array of indices
    
    # Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 110.0
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        # Vectorized greedy selection
        # counts[i] = sum(uncovered[neighbors[i]])
        # This is hard to vectorize fully without memory blowup, but N=6561 is small.
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
            
    # Local Search
    # Helper to compute coverage counts
    def get_coverage_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    # Helper to check if code is valid (all covered)
    def is_valid(code):
        counts = get_coverage_counts(code)
        return np.all(counts >= 1)

    # Helper to remove redundant codewords
    def reduce_code(code):
        if not code:
            return code
        counts = get_coverage_counts(code)
        new_code = []
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in code:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            code = next_code
        return code

    C = reduce_code(C)
    current_size = len(C)
    best_size = current_size
    best_C = C.copy()
    
    # Precompute in_code set for fast lookup
    in_code = set(C)
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Reduction Pass
        new_C = reduce_code(C)
        if len(new_C) < current_size:
            C = new_C
            current_size = len(C)
            in_code = set(C)
            if current_size < best_size:
                best_size = current_size
                best_C = C.copy()
            improved = True
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        # Calculate coverage counts once
        counts = get_coverage_counts(C)
        
        # For each codeword, try to find a swap
        # To avoid O(N^2) behavior, we limit the number of swap attempts per iteration
        attempts = 0
        max_attempts = 1000 # Arbitrary limit per outer loop iteration
        
        for c in C:
            if attempts > max_attempts:
                break
            if time.time() - start_time > time_limit:
                break
                
            # Find unique points covered by c
            # u is in neighbors[c] and counts[u] == 1
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # Should have been removed
                continue
                
            # Find candidates c' that cover all unique_points
            # c' must be in intersection of neighbors[u] for u in unique_points
            # Since unique_points is small, we can iterate through neighbors of the first unique point
            # and check if they cover the rest
            
            # Optimization: If len(unique_points) > 3, it's unlikely a single c' covers them all
            # because the intersection shrinks rapidly.
            if len(unique_points) > 3:
                continue
                
            # Get neighbors of the first unique point
            first_u = unique_points[0]
            candidates = neighbors[first_u]
            
            found_swap = False
            for c_prime in candidates:
                if c_prime in in_code:
                    continue
                
                # Check if c_prime covers all unique_points
                valid = True
                for u in unique_points:
                    if u not in neighbors[c_prime]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try the swap
                # Remove c, add c_prime
                C.remove(c)
                in_code.remove(c)
                C.append(c_prime)
                in_code.add(c_prime)
                
                # Check if this swap allows further reduction
                reduced_C = reduce_code(C)
                
                if len(reduced_C) < current_size:
                    # Accept swap and reduction
                    C = reduced_C
                    current_size = len(C)
                    in_code = set(C)
                    if current_size < best_size:
                        best_size = current_size
                        best_C = C.copy()
                    improved = True
                    found_swap = True
                    break
                else:
                    # Revert swap
                    C.remove(c_prime)
                    in_code.remove(c_prime)
                    C.append(c)
                    in_code.add(c)
                    # We can break here because other candidates from this 'c' 
                    # are likely to have similar issues, or we just move to next c
                    break 
                    
            if found_swap:
                # Restart the loop over C as it has changed
                break
            
        if not improved and not found_swap:
            # If no improvement, we might be stuck. 
            # We can add some randomness here or just break if we've tried many times.
            # For simplicity, we break if no improvement in a full pass.
            # To avoid infinite loop, we can track iterations.
            # But the time limit is the main guard.
            pass

    # Final Reduction
    best_C = reduce_code(best_C)
    
    return [list(all_words[i]) for i in best_C]
