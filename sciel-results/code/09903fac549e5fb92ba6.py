import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 295

    # Precompute points
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2).astype(np.int32)
    
    def can_add(p, S_arr):
        """Check if point p can be added to set S (numpy array of indices). Returns True/False."""
        if len(S_arr) == 0:
            return True
        
        # 1. Distances from p to all in S must be distinct
        d_p_S = D[p, S_arr]
        if len(np.unique(d_p_S)) != len(S_arr):
            return False
        
        # 2. For each s in S, d(p,s) must not equal d(s,t) for any t in S
        d_S_S = D[np.ix_(S_arr, S_arr)]
        # d_p_S[i] is distance from p to S_arr[i]
        # Check if d_p_S[i] appears in row i of d_S_S
        matches = (d_S_S == d_p_S[:, np.newaxis])
        if matches.any():
            return False
        
        return True

    def can_add_fast(p, S_arr, d_p_S_cache=None):
        """Faster can_add using precomputed d_p_S if available."""
        if len(S_arr) == 0:
            return True, None
        
        d_p_S = D[p, S_arr]
        if len(np.unique(d_p_S)) != len(S_arr):
            return False, d_p_S
        
        d_S_S = D[np.ix_(S_arr, S_arr)]
        matches = (d_S_S == d_p_S[:, np.newaxis])
        if matches.any():
            return False, d_p_S
        
        return True, d_p_S

    best_S = []
    best_size = 0
    
    # Phase 1: Multiple greedy restarts with different random orders
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        S = []
        candidates = list(range(M))
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline - 10:
                break
            if can_add(p, np.array(S, dtype=np.int32)):
                S.append(p)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    # Phase 2: Iterated Local Search (ILS)
    # Perturb best solution, then try to improve
    current_S = best_S.copy()
    current_size = best_size
    
    for iteration in range(2000):
        if time.time() > deadline - 10:
            break
        
        # Perturbation: remove k random points
        if current_size == 0:
            break
        
        k = random.randint(1, min(5, current_size))
        to_remove = random.sample(current_S, k)
        remove_set = set(to_remove)
        S = [p for p in current_S if p not in remove_set]
        
        # Try to add points back / new points
        candidates = [p for p in range(M) if p not in set(S)]
        random.shuffle(candidates)
        
        S_arr = np.array(S, dtype=np.int32)
        for p in candidates:
            if time.time() > deadline - 10:
                break
            if can_add(p, S_arr):
                S.append(p)
                S_arr = np.array(S, dtype=np.int32)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            current_S = S.copy()
            current_size = best_size
        else:
            # Accept with some probability (simulated annealing style)
            # or just keep current best
            if random.random() < 0.1:
                current_S = S.copy()
                current_size = len(S)
    
    # Phase 3: More aggressive local search - single point swaps
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        
        if best_size == 0:
            break
        
        S = best_S.copy()
        S_set = set(S)
        S_arr = np.array(S, dtype=np.int32)
        
        # Try swapping one point out and one in
        if len(S) > 0:
            p_out = random.choice(S)
            S.remove(p_out)
            S_arr = np.array(S, dtype=np.int32)
            
            # Try adding random points to fill the gap
            candidates = [p for p in range(M) if p not in S_set and p != p_out]
            random.shuffle(candidates)
            
            improved = False
            for p_in in candidates[:200]:  # Limit to first 200 for speed
                if time.time() > deadline - 5:
                    break
                if can_add(p_in, S_arr):
                    S.append(p_in)
                    improved = True
                    break
            
            if improved and len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
    
    return [tuple(points[i]) for i in best_S]
