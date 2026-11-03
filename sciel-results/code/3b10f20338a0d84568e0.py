import time
import random
import math
import numpy as np

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neighbors = [[] for _ in range(total)]
    for idx in range(total):
        neighbors[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neighbors[idx].append(new_idx)
    return neighbors, total

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    neighbors, _ = _precompute_neighbors(q, n, radius)
    
    # Precompute sets for fast set operations
    neighbor_sets = [set(neighbors[i]) for i in range(total)]
    
    # Precompute dist-2 neighbors for packing
    neigh2 = [set() for _ in range(total)]
    for u in range(total):
        for v in neighbors[u]:
            neigh2[u].update(neighbors[v])

    # Ball matrix for greedy gains
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        for u in neighbors[c]:
            ball_matrix[c, u] = 1

    def greedy_init_packing(seed):
        r = random.Random(seed)
        
        # 1. Greedy Packing (Max Independent Set in G2)
        available_mask = np.ones(total, dtype=bool)
        code = []
        
        while available_mask.any():
            avail_indices = np.where(available_mask)[0]
            if len(avail_indices) == 0: break
            # Pick random
            c = int(avail_indices[r.randint(0, len(avail_indices)-1)])
            code.append(c)
            # Remove c and its G2 neighbors
            for u in neigh2[c]:
                available_mask[u] = False
        
        # 2. Greedy Cover of Holes
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            for u in neighbors[c]:
                counts[u] += 1
        
        uncovered_mask = (counts == 0).astype(np.uint8)
        
        while uncovered_mask.any():
            gains = ball_matrix @ uncovered_mask
            max_gain = int(gains.max())
            if max_gain == 0: break
            candidates = np.where(gains == max_gain)[0]
            c = int(candidates[r.randint(0, len(candidates)-1)])
            code.append(c)
            counts += ball_matrix[c]
            uncovered_mask = (counts == 0).astype(np.uint8)
            
            if time.monotonic() > deadline - 5:
                break
                
        return code, counts

    def polish(code, counts, time_budget):
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget
        
        def is_removable(c):
            return np.min(counts[neighbors[c]]) > 1

        # 1. Remove redundant
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    code_set.discard(c)
                    counts -= ball_matrix[c]
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]

        # 2. Simulated Annealing
        temp = 5.0
        cool_rate = 0.99
        min_temp = 0.1
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.4:
                # Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    # Count holes if removed
                    holes = 0
                    for u in neighbors[c]:
                        if counts[u] == 1:
                            holes += 1
                            
                    if holes == 0:
                        code.pop(idx)
                        code_set.discard(c)
                        counts -= ball_matrix[c]
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                    else:
                        if rng.random() < math.exp(-holes / temp):
                            code.pop(idx)
                            code_set.discard(c)
                            counts -= ball_matrix[c]
                            
            elif move_type < 0.8:
                # Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check validity:
                    # Words in ball[old_c] but not in ball[new_c] lose 1 count.
                    # If any has count 1, it becomes 0.
                    old_ball_set = neighbor_sets[old_c]
                    new_ball_set = neighbor_sets[new_c]
                    
                    holes = 0
                    for u in old_ball_set:
                        if u not in new_ball_set:
                            if counts[u] == 1:
                                holes += 1
                                if holes > 5: break # Early exit for SA
                    
                    if holes == 0:
                        # Valid swap
                        counts -= ball_matrix[old_c]
                        counts += ball_matrix[new_c]
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                    else:
                        if rng.random() < math.exp(-holes / temp):
                            counts -= ball_matrix[old_c]
                            counts += ball_matrix[new_c]
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
            
            else:
                # Add
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        counts += ball_matrix[new_c]
            
            temp *= cool_rate

        # Final Remove
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    code_set.discard(c)
                    counts -= ball_matrix[c]
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]

        return best_code, best_size

    overall_best_code = None
    overall_best_size = total + 1
    
    num_restarts = 40
    time_per_restart = (deadline - time.monotonic()) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 5:
            break
        
        code, counts = greedy_init_packing(42 + restart * 13)
        
        if not code:
            continue
        
        polished_code, polished_size = polish(code, counts, time_per_restart - 0.5)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
            
    if overall_best_code is None:
        code, counts = greedy_init_packing(42)
        overall_best_code = code
        overall_best_size = len(code)
        
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result

_neigh_sets_cache = {}

def _get_neigh_sets(q, n, radius):
    key = (q, n, radius)
    if key not in _neigh_sets_cache:
        total = q ** n
        neigh, _ = _precompute_neighbors(q, n, radius)
        _neigh_sets_cache[key] = [set(neigh[i]) for i in range(total)]
    return _neigh_sets_cache[key]
