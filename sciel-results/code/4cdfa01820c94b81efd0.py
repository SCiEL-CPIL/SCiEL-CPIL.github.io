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
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit:
                break
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

    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors as numpy arrays
    neighbors_np = []
    for i, w in enumerate(all_words):
        neigh_set = {i}
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors_np.append(np.array(list(neigh_set), dtype=np.int32))

    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 105.0

    while np.any(uncovered):
        # Vectorized-ish greedy: pick word covering most uncovered
        # To speed this up, we can sample or use a better heuristic.
        # But for 6561 items, a full scan is ~100ms if done in C.
        # In Python, a full scan is slow.
        
        # Optimization: Instead of checking all 6561 words every time,
        # we can maintain a priority or just accept a slightly suboptimal greedy.
        # However, a standard trick for small N is just to do the scan but optimized.
        
        # Let's try to vectorize the count calculation.
        # counts = np.sum(uncovered[neighbors_np], axis=1) -> This requires a 2D array.
        # Let's build a 2D neighbors array once.
        
        # Rebuild neighbors as a 2D padded array for vectorized sum?
        # Each has 17 neighbors.
        
        best_idx = -1
        best_count = -1
        
        # Fast loop with local vars
        unc = uncovered
        nbrs = neighbors_np
        for i in range(N):
            cnt = np.count_nonzero(unc[nbrs[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                if best_count == 17: # Max possible
                    break
        
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[nbrs[best_idx]] = False
        if time.time() - start_time > time_limit:
            break

    # 2. Local Search
    def get_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbors_np[c]] += 1
        return counts

    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                # Check if c has any unique points (count == 1)
                # If min count >= 2, all are covered by others, so remove c.
                if np.min(current_counts[neighbors_np[c]]) >= 2:
                    current_counts[neighbors_np[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()

    def find_candidates(points):
        if not points:
            return set(range(N))
        cand_set = set(neighbors_np[points[0]].tolist())
        for p in points[1:]:
            cand_set &= set(neighbors_np[p].tolist())
            if not cand_set:
                break
        return cand_set

    # Local Search Loop
    while time.time() - start_time < time_limit:
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        if len(C) == old_size:
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                u_mask = counts[neighbors_np[c]] == 1
                unique_points = neighbors_np[c][u_mask]
                
                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors_np[c]] -= 1
                    improved = True
                    break
                
                if len(unique_points) > 2:
                    continue
                
                candidates_set = find_candidates(unique_points.tolist())
                c_set = set(C)
                candidates_set -= c_set
                
                if not candidates_set:
                    continue
                    
                c_prime = random.choice(list(candidates_set))
                
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                counts[neighbors_np[c]] -= 1
                counts[neighbors_np[c_prime]] += 1
                
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
                    # Keep the swap if it's neutral to explore
                    C = new_C
                    counts = new_counts
                    improved = True
                    break
            
            if not improved:
                # Kick: random swap to escape local min
                for _ in range(5):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    c_prime = neighbors_np[c][random.randint(0, 16)]
                    if c_prime in C: continue
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors_np[c]] -= 1
                    counts[neighbors_np[c_prime]] += 1
                    
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

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
