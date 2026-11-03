import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    D = 11
    
    # Generate all candidate vectors with norm^2 = 4
    candidates = []
    
    # Type A: one coordinate is +/-2
    for i in range(D):
        v = [0] * D
        v[i] = 2
        candidates.append(tuple(v))
        v[i] = -2
        candidates.append(tuple(v))
    
    # Type D: four coordinates are +/-1
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(tuple(v))
    
    N = len(candidates)
    candidates_arr = np.array(candidates, dtype=np.int8)
    
    # Build conflict matrix
    conflict = np.zeros((N, N), dtype=bool)
    for i in range(N):
        if time.time() > deadline - 60:
            break
        dots = candidates_arr @ candidates_arr[i]
        conflict[i, i+1:] = dots[i+1:] > 2
        conflict[i+1:, i] = dots[i+1:] > 2
    
    degrees = np.sum(conflict, axis=1)
    
    best_set = set()
    best_size = 0
    
    # Multiple greedy strategies
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 40:
            break
        
        if restart % 3 == 0:
            order = np.argsort(degrees)
        elif restart % 3 == 1:
            order = np.random.permutation(N)
        else:
            # Weighted random
            weights = 1.0 / (degrees + 1)
            weights /= weights.sum()
            order = np.random.choice(N, size=N, replace=False, p=weights)
        
        current_set = set()
        for idx in order:
            if current_set:
                if np.any(conflict[idx, list(current_set)]):
                    continue
            current_set.add(idx)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Local search: 1-swap
    if time.time() < deadline - 15:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in list(best_set):
                best_set.discard(i)
                current_list = list(best_set)
                for j in range(N):
                    if j in best_set:
                        continue
                    if not np.any(conflict[j, current_list]):
                        best_set.add(j)
                        if len(best_set) > best_size:
                            best_size = len(best_set)
                            improved = True
                        best_set.discard(j)
                        break
                if not improved:
                    best_set.add(i)
                else:
                    break
            if not improved:
                break
    
    result = candidates_arr[list(best_set)]
    return result
