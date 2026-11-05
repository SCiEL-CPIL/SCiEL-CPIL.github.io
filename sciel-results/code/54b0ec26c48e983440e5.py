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
        
    # Helper: Compute coverage counts for a set of codewords
    def compute_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    # Helper: Reduce code by removing redundant words
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

    # Helper: Local Search (Swap-based)
    def local_search(code, counts, time_budget):
        C = list(code)
        counts = counts.copy()
        start_time = time.time()
        
        # Initial reduction
        C, counts = reduce_code(C, counts)
        best_size = len(C)
        best_C = C.copy()
        best_counts = counts.copy()
        
        while time.time() - start_time < time_budget:
            # 1. Try to reduce
            old_size = len(C)
            C, counts = reduce_code(C, counts)
            
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()
                
            if time.time() - start_time > time_budget:
                break
                
            # 2. If no reduction, try swaps
            if len(C) == old_size:
                improved = False
                # Shuffle to randomize
                indices = list(range(len(C)))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() - start_time > time_budget:
                        break
                    c = C[idx]
                    u_indices = neighbors[c]
                    unique_mask = counts[u_indices] == 1
                    unique_points = u_indices[unique_mask]
                    
                    if len(unique_points) == 0:
                        # Force removal
                        C.remove(c)
                        counts[neighbors[c]] -= 1
                        improved = True
                        break
                    
                    if len(unique_points) > 3:
                        continue
                    
                    # Find candidates covering unique_points
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
                    
                    candidates_set.difference_update(C)
                    if not candidates_set:
                        continue
                        
                    c_prime = random.choice(list(candidates_set))
                    
                    # Apply swap
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
                        # Accept neutral swap to escape local minimum
                        C = new_C
                        counts = new_counts
                        improved = True
                        break
            
            if not improved:
                # Kick: random swap
                for _ in range(5):
                    if time.time() - start_time > time_budget: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C: continue
                    
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

    # Main Loop: Multi-Start
    global_best_C = []
    global_best_counts = None
    global_best_size = float('inf')
    
    start_time = time.time()
    time_limit = 115.0
    num_restarts = 5 # Adjust based on timing
    
    for i in range(num_restarts):
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break
            
        # Randomized Greedy Initialization
        uncovered = np.ones(N, dtype=bool)
        C = []
        
        # To make it faster, we can precompute a list of indices
        indices = list(range(N))
        
        while np.any(uncovered):
            # Find top 10 candidates
            # Scanning all N is fast enough (~6561)
            best_idx = -1
            best_count = -1
            second_best_idx = -1
            second_best_count = -1
            # For simplicity and speed, just pick one, but randomize tie-breaking
            # Or pick from top 3
            top_indices = []
            top_counts = []
            
            # Optimization: Instead of scanning all N every time, 
            # we can just scan. 6561 * 500 is 3M. 5 restarts is 15M. 
            # Python loop is slow. 
            # Let's stick to the previous method but maybe vectorize the count?
            # np.sum(uncovered[neighbors[i]]) is fast.
            
            # We'll just do the standard scan but add a bit of randomness if tied
            # Actually, just picking the best is deterministic. 
            # To get different starts, we can randomly shuffle the order of indices 
            # or pick randomly among those with max count.
            
            max_count = -1
            candidates = []
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > max_count:
                    max_count = cnt
                    candidates = [i]
                elif cnt == max_count:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Pick randomly from candidates to diversify
            best_idx = random.choice(candidates)
            
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
            # Time check inside greedy? Greedy is fast.
        
        # Reduce and Local Search
        counts = compute_counts(C)
        C, counts = local_search(C, counts, time_budget=(time_limit - (time.time() - start_time)) / (num_restarts - i))
        
        if len(C) < global_best_size:
            global_best_size = len(C)
            global_best_C = C.copy()
            global_best_counts = counts.copy()
            
    # Final reduction on best
    if global_best_C:
        global_best_C, _ = reduce_code(global_best_C, compute_counts(global_best_C))
        
    return [list(all_words[i]) for i in global_best_C]
