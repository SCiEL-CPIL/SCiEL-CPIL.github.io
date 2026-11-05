import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
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
    
    # Precompute neighbors as a list of lists for fast access in local search
    # And a boolean matrix for fast greedy
    neighbors = []
    M = np.zeros((N, N), dtype=bool)
    
    for i, w in enumerate(all_words):
        neigh_set = [i]
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                idx = word_to_idx[new_w]
                neigh_set.append(idx)
                M[i, idx] = True
        # Also set M[idx, i] = True? 
        # Distance is symmetric. 
        # But M is used as M @ uncovered. 
        # M[i, j] should be 1 if i covers j.
        # So we need M[i, j] = 1 if dist(i, j) <= 1.
        # The loop above sets M[i, idx] = 1.
        # We also need M[idx, i] = 1? 
        # If i covers idx, does idx cover i? Yes, dist is symmetric.
        # So M should be symmetric.
        # We can just fill M[i, neigh] = True and M[neigh, i] = True.
        # Or just fill M[i, neigh] and rely on symmetry? 
        # M @ uncovered: 
        # counts[i] = sum_j M[i, j] * uncovered[j]
        # This is number of uncovered words covered by i.
        # This is correct.
        # We need M[i, j] to be True if i covers j.
        # My loop: for each i, for each neighbor idx of i, set M[i, idx] = True.
        # This is sufficient.
        neighbors.append(neigh_set)
        
    # Ensure symmetry for correctness if we use it that way, but M @ uncovered only needs row i.
    # However, for local search, we need neighbors[c].
    
    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    while np.any(uncovered):
        # Vectorized max coverage
        # counts[i] = sum(uncovered[j] for j in neighbors[i])
        # This is M @ uncovered
        counts = M @ uncovered
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
            
    # 2. Local Search
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                # Check if c is redundant
                # c is redundant if all neighbors of c are covered by at least 2 codewords
                # i.e., min(current_counts[neighbors[c]]) >= 2
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        improved = False
        # Iterate through C to find a candidate for swap
        # We prioritize codewords with few unique points
        # To avoid O(N^2) in the swap search, we limit the scope or use heuristics
        
        # Find codeword with minimal unique points
        min_unique_count = N
        target_c = -1
        target_unique = []
        
        for c in C:
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            if len(unique_points) < min_unique_count:
                min_unique_count = len(unique_points)
                target_c = c
                target_unique = unique_points.tolist()
                
        if target_c == -1 or min_unique_count == 0:
            continue # Should have been reduced
            
        if min_unique_count > 3:
            continue # Unlikely to find a single replacement
            
        # Find candidates c' that cover all target_unique
        if len(target_unique) == 1:
            candidates_set = set(neighbors[target_unique[0]].tolist())
        else:
            candidates_set = set(neighbors[target_unique[0]].tolist())
            for u in target_unique[1:]:
                candidates_set &= set(neighbors[u].tolist())
                if not candidates_set:
                    break
        
        if not candidates_set:
            continue
        
        # Filter candidates: must not be in C
        candidates_set.difference_update(C)
        
        if not candidates_set:
            continue
            
        # Pick a candidate
        # Heuristic: Pick one that maximizes coverage of currently covered points?
        # Or just pick one that allows reduction.
        # We try all candidates if few, otherwise pick one randomly or by heuristic.
        
        # To keep it fast, we only try if candidates_set is small, or pick best heuristic.
        # Let's pick the one with max overlap with C (to create redundancy)
        best_c_prime = None
        max_overlap = -1
        
        # Limit iteration for speed
        for c_prime in list(candidates_set):
            # Calculate overlap with C
            # Overlap = sum(counts[neighbors[c_prime]])? No.
            # We want c_prime to cover points already covered by C to enable removal of others.
            # So we want c_prime to have high count in neighbors.
            # counts[neighbors[c_prime]] gives the current coverage of neighbors of c_prime.
            # We want to maximize this sum?
            # If c_prime covers many points that are already covered by 2+, it creates more redundancy.
            overlap = np.sum(counts[neighbors[c_prime]])
            if overlap > max_overlap:
                max_overlap = overlap
                best_c_prime = c_prime
                
        if best_c_prime is None:
            continue
            
        # Apply swap
        old_C = C.copy()
        old_counts = counts.copy()
        
        C.remove(target_c)
        C.append(best_c_prime)
        
        counts[neighbors[target_c]] -= 1
        counts[neighbors[best_c_prime]] += 1
        
        # Check validity
        if np.any(counts < 1):
            C = old_C
            counts = old_counts
            continue
            
        # Try to reduce
        new_C, new_counts = reduce_code(C, counts)
        
        if len(new_C) < len(old_C):
            # Improvement!
            C = new_C
            counts = new_counts
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
            improved = True
            break # Restart loop
            
        else:
            # No immediate reduction.
            # Revert to avoid cycling, or keep with small probability?
            # For determinism and speed, revert.
            C = old_C
            counts = old_counts

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
