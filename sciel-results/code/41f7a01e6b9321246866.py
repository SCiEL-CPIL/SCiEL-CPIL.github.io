import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
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
    
    # Precompute neighbors as numpy arrays
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
    
    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(N)]
    
    start_time = time.time()
    time_limit = 115.0
    
    # Fast greedy initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Use a priority-based approach: maintain count of uncovered neighbors for each candidate
    # For speed, we scan all 6561 words each iteration but use numpy
    while np.any(uncovered):
        if time.time() - start_time > time_limit:
            break
        # Compute coverage count for each word
        # uncovered[neighbors[i]] gives the uncovered points covered by word i
        # We want the word that covers the most uncovered points
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
    
    # Local search
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
        
        if time.time() - start_time > time_limit:
            break
        
        # 2. Swap Pass
        if len(C) == old_size:
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
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
                    candidates_set = neighbor_sets[unique_points[0]].copy()
                else:
                    candidates_set = neighbor_sets[unique_points[0]].copy()
                    for u in unique_points[1:]:
                        candidates_set &= neighbor_sets[u]
                        if not candidates_set:
                            break
                
                if not candidates_set:
                    continue
                
                candidates_set.difference_update(C)
                
                if not candidates_set:
                    continue
                
                cand_list = list(candidates_set)
                random.shuffle(cand_list)
                
                for c_prime in cand_list:
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
                        improved = True
                        break
                    else:
                        # Accept neutral swap with some probability to escape local minima
                        if random.random() < 0.1:
                            C = new_C
                            counts = new_counts
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                # Random kick
                for _ in range(3):
                    if time.time() - start_time > time_limit:
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
                        if random.random() < 0.05:
                            C = new_C
                            counts = new_counts

    # Final reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
