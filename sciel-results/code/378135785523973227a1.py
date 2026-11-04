import time
import math
import numpy as np
from itertools import combinations

def _score(U):
    """Compute the score for a given set U (as a sorted list) using vectorized ops."""
    U_arr = np.array(U, dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr.max())
    if max_u == 0:
        return 1.0  # degenerate case
    
    # Compute U+U and U-U using broadcasting
    sum_set = np.unique(U_arr[:, None] + U_arr[None, :])
    diff_set = np.unique(U_arr[:, None] - U_arr[None, :])
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 1.0
    ratio = n_diff / n_sum
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _search_small_sets(deadline):
    """Search over small sets of nonnegative integers containing 0."""
    best_score = 0.0
    best_set = [0]
    
    max_val = 30
    max_size = 5
    
    candidates = list(range(1, max_val + 1))
    
    for size in range(1, max_size + 1):
        if time.time() > deadline:
            break
        for combo in combinations(candidates, size):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            U.sort()
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score

def _local_search(U, deadline, max_iter=500):
    """Local search: try adding/removing/replacing elements near existing ones."""
    U_set = set(U)
    best_score = _score(U)
    best_U = list(U)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbors
        neighbors = []
        
        # Try adding a new element near existing ones
        for u in U_set:
            for delta in [-3, -2, -1, 1, 2, 3, 5, 7, 10]:
                new_val = u + delta
                if new_val > 0 and new_val not in U_set:
                    new_U = sorted(list(U_set) + [new_val])
                    neighbors.append(new_U)
        
        # Try replacing an element
        for u in U_set:
            if u == 0:
                continue
            for delta in [-3, -2, -1, 1, 2, 3, 5, 7, 10]:
                new_val = u + delta
                if new_val > 0 and new_val not in U_set:
                    new_U = sorted((U_set - {u}) | {new_val})
                    neighbors.append(new_U)
        
        # Try geometric scaling
        if len(U_set) >= 2:
            for base in [2, 3, 4]:
                scaled = sorted([int(round(u * base)) for u in U_set if u > 0] + [0])
                if len(set(scaled)) == len(U_set):
                    neighbors.append(scaled)
        
        if not neighbors:
            break
        
        # Evaluate all neighbors, keep best
        best_neighbor = None
        best_neighbor_score = best_score
        
        for nU in neighbors:
            if time.time() > deadline:
                break
            s = _score(nU)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = nU
        
        if best_neighbor is None:
            break
        
        best_U = best_neighbor
        best_score = best_neighbor_score
        U_set = set(best_U)
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Phase 1: Exhaustive search over small sets
    best_set, best_score = _search_small_sets(deadline)
    
    # Phase 2: Try geometric progressions
    for B in [2, 3, 4, 5, 6, 7, 8]:
        for k in range(1, 12):
            if time.time() > deadline:
                break
            U = [B**i for i in range(k+1)]
            if max(U) > 100000:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Phase 3: Try mixed arithmetic-geometric sets
    for base in [2, 3, 4]:
        for offset in [0, 1, 2]:
            for k in range(2, 8):
                if time.time() > deadline:
                    break
                U = sorted([(base**i + offset) for i in range(k) if base**i + offset > 0] + [0])
                if len(set(U)) < 3:
                    continue
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Phase 4: Local search from best set
    if time.time() < deadline - 10:
        best_set, best_score = _local_search(best_set, deadline)
    
    # Phase 5: Second round of local search with more iterations if time allows
    if time.time() < deadline - 5:
        best_set, best_score = _local_search(best_set, deadline, max_iter=1000)
    
    return best_set
