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
    
    # Precompute neighbors as numpy arrays for fast operations
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
    
    # Precompute neighbor sets as Python sets for fast intersection
    neighbor_sets = [set(arr.tolist()) for arr in neighbors]
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts
    
    def reduce_code(C, counts):
        """Peel: iteratively remove codewords that are redundant."""
        if not C:
            return C, counts
        current_C = list(C)
        current_counts = counts.copy()
        while True:
            removed_any = False
            next_C = []
            for c in current_C:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_C.append(c)
            current_C = next_C
            if not removed_any:
                break
        return current_C, current_counts
    
    def local_search_tight(C, counts, time_budget):
        """Tightly coupled reduction + swap local search."""
        best_C = C.copy()
        best_counts = counts.copy()
        best_size = len(C)
        local_start = time.time()
        
        while time.time() - local_start < time_budget:
            # Pass 1: Reduction
            old_size = len(C)
            C, counts = reduce_code(C, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
            
            if time.time() - local_start > time_budget:
                break
            
            # Pass 2: Swap if no reduction
            if len(C) == old_size:
                improved = False
                indices = list(range(len(C)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() - local_start > time_budget:
                        break
                    c = C[idx]
                    u_indices = neighbors[c]
                    unique_mask = counts[u_indices] == 1
                    unique_points = u_indices[unique_mask]
                    
                    if len(unique_points) == 0:
                        C.remove(c)
                        counts[neighbors[c]] -= 1
                        improved = True
                        break
                    
                    if len(unique_points) > 3:
                        continue
                    
                    # Find candidates that cover all unique points
                    if len(unique_points) == 1:
                        candidates_set = set(neighbor_sets[unique_points[0]])
                    else:
                        candidates_set = set(neighbor_sets[unique_points[0]])
                        for u in unique_points[1:]:
                            candidates_set &= neighbor_sets[u]
                            if not candidates_set:
                                break
                    
                    if not candidates_set:
                        continue
                    
                    candidates_set.difference_update(C)
                    
                    if not candidates_set:
                        continue
                    
                    # Pick candidate with max overlap
                    best_c_prime = None
                    best_overlap = -1
                    for cp in list(candidates_set)[:20]:  # Limit to 20 for speed
                        overlap = np.sum(counts[neighbors[cp]] >= 2)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_c_prime = cp
                    
                    if best_c_prime is None:
                        best_c_prime = random.choice(list(candidates_set))
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(best_c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[best_c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                    
                    # Try to reduce
                    new_C, new_counts = reduce_code(C, counts)
                    
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        improved = True
                        break
                    else:
                        # Keep swap to escape local minimum
                        C = new_C
                        counts = new_counts
                        improved = True
                        break
                
                if not improved:
                    # Random kicks
                    for _ in range(3):
                        if time.time() - local_start > time_budget:
                            break
                        if not C:
                            break
                        c_idx = random.randint(0, len(C) - 1)
                        c = C[c_idx]
                        cand = neighbors[c]
                        c_prime = cand[random.randint(0, len(cand) - 1)]
                        if c_prime in C:
                            continue
                        
                        old_C = C.copy()
                        old_counts = counts.copy()
                        
                        C.remove(c)
                        C.append(c_prime)
                        counts[neighbors[c]] -= 1
                        counts[neighbors[c_prime]] += 1
                        
                        if np.any(counts < 1):
                            C = old_C
                            counts = old_counts
                            continue
                        
                        new_C, new_counts = reduce_code(C, counts)
                        
                        if len(new_C) < len(old_C):
                            C = new_C
                            counts = new_counts
                            if len(C) < best_size:
                                best_size = len(C)
                                best_C = C.copy()
                                best_counts = counts.copy()
                            break
                        else:
                            C = new_C
                            counts = new_counts
        
        return best_C, best_counts
    
    def polish_pass(C, counts, time_budget):
        """Final high-precision polish with more aggressive swaps."""
        best_C = C.copy()
        best_counts = counts.copy()
        best_size = len(C)
        local_start = time.time()
        
        while time.time() - local_start < time_budget:
            C, counts = reduce_code(C, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
            
            if time.time() - local_start > time_budget:
                break
            
            # More aggressive: try swaps with up to 4 unique points
            indices = list(range(len(C)))
            random.shuffle(indices)
            improved = False
            
            for idx in indices:
                if time.time() - local_start > time_budget:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]
                
                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 4:
                    continue
                
                if len(unique_points) == 1:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                else:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                    for u in unique_points[1:]:
                        candidates_set &= neighbor_sets[u]
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                
                # Try all candidates if few, else sample
                cand_list = list(candidates_set)
                if len(cand_list) > 10:
                    random.shuffle(cand_list)
                    cand_list = cand_list[:10]
                
                for cp in cand_list:
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(cp)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[cp]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                    
                    new_C, new_counts = reduce_code(C, counts)
                    
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        improved = True
                        break
                    else:
                        C = new_C
                        counts = new_counts
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_C, best_counts
    
    # Main loop
    best_C = None
    best_counts = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # First run: deterministic greedy
    uncovered = np.ones(N, dtype=bool)
    C = []
    while np.any(uncovered) and time.time() - start_time < time_limit:
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
    
    counts = get_counts(C)
    
    # Ensure coverage
    if np.any(counts < 1):
        while np.any(counts < 1) and time.time() - start_time < time_limit:
            uncovered_mask = counts < 1
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered_mask[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            counts[neighbors[best_idx]] += 1
    
    C, counts = reduce_code(C, counts)
    
    if len(C) < best_size:
        best_size = len(C)
        best_C = C.copy()
        best_counts = counts.copy()
    
    # Local search on first solution
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 15:
        best_C, best_counts = local_search_tight(best_C, best_counts, remaining_time * 0.7)
        if len(best_C) < best_size:
            best_size = len(best_C)
    
    # Multiple randomized restarts
    remaining_time = time_limit - (time.time() - start_time)
    num_restarts = min(5, max(1, int(remaining_time / 20)))
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Randomized greedy
        uncovered = np.ones(N, dtype=bool)
        C = []
        while np.any(uncovered) and time.time() - start_time < time_limit - 10:
            counts_arr = np.array([np.sum(uncovered[neighbors[i]]) for i in range(N)])
            if counts_arr.max() == 0:
                break
            max_count = counts_arr.max()
            max_indices = np.where(counts_arr == max_count)[0]
            best_idx = max_indices[rng.randint(0, len(max_indices) - 1)]
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
        
        if not C:
            continue
        
        counts = get_counts(C)
        
        if np.any(counts < 1):
            while np.any(counts < 1) and time.time() - start_time < time_limit - 10:
                uncovered_mask = counts < 1
                best_idx = -1
                best_count = -1
                for i in range(N):
                    cnt = np.sum(uncovered_mask[neighbors[i]])
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                if best_count == 0:
                    break
                C.append(best_idx)
                counts[neighbors[best_idx]] += 1
        
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 15 and len(C) <= best_size + 3:
            C, counts = local_search_tight(C, counts, remaining_time * 0.5)
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
    
    # Final polish pass
    remaining_time = time_limit - (time.time() - start_time)
    if best_C is not None and remaining_time > 5:
        best_C, best_counts = polish_pass(best_C, best_counts, remaining_time * 0.8)
        if len(best_C) < best_size:
            best_size = len(best_C)
    
    # Final reduction
    if best_C is not None:
        final_counts = get_counts(best_C)
        best_C, final_counts = reduce_code(best_C, final_counts)
    
    if best_C is None:
        best_C = list(range(N))
    
    return [list(all_words[i]) for i in best_C]
