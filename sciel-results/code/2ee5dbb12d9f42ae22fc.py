"""No-isosceles set: construct a set of grid points with no three forming an isosceles triangle."""
import numpy as np
import time
import random

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 280  # leave 20s safety margin

# Pre-compute all grid points as (x, y)
_all_points = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)

def _squared_dist_matrix(S):
    """Compute pairwise squared distances for array S of shape (K, 2)."""
    K = len(S)
    if K == 0:
        return np.empty((0, 0))
    diff = S[:, np.newaxis, :] - S[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)

def _check_valid(S):
    """Check if set S (array of points, shape (K,2)) has no isosceles triangles."""
    K = len(S)
    if K <= 2:
        return True
    D = _squared_dist_matrix(S)
    # For each point k, check if any distance from k to others is repeated
    for k in range(K):
        dists = D[k, :].copy()
        dists[k] = -1
        if len(set(dists.tolist())) < len(dists):
            return False
    return True

def _greedy_with_order(order):
    """Greedy construction using a specific order of point indices."""
    S = []
    S_arr = np.empty((0, 2), dtype=np.int32)
    for idx in order:
        p = _all_points[idx].reshape(1, 2)
        if len(S_arr) == 0:
            S_arr = p
        else:
            S_arr = np.vstack([S_arr, p])
        if _check_valid(S_arr):
            S.append(idx)
        else:
            # Revert
            S_arr = S_arr[:-1]
    return S

def _local_search(S, time_limit=5.0):
    """Improve set S via single additions and remove-1-add-2 swaps."""
    S_set = set(S)
    S_arr = np.array([_all_points[i] for i in S], dtype=np.int32) if S else np.empty((0, 2), dtype=np.int32)
    best_score = len(S_set)
    best_S = list(S_set)
    start = time.time()
    
    # Phase 1: Try single additions
    for idx in range(NUM_POINTS):
        if time.time() - start > time_limit:
            break
        if idx in S_set:
            continue
        p = _all_points[idx].reshape(1, 2)
        if len(S_arr) == 0:
            new_S = p
        else:
            new_S = np.vstack([S_arr, p])
        if _check_valid(new_S):
            S_set.add(idx)
            S_arr = new_S
            best_S = list(S_set)
            best_score = len(S_set)
    
    # Phase 2: Try remove-1-add-2 swaps
    while time.time() - start < time_limit:
        improved = False
        S_list = list(S_set)
        for i in range(len(S_list)):
            if time.time() - start > time_limit:
                break
            S_without_i = S_list[:i] + S_list[i+1:]
            S_without_i_set = set(S_without_i)
            S_without_i_arr = np.array([_all_points[j] for j in S_without_i], dtype=np.int32) if S_without_i else np.empty((0, 2), dtype=np.int32)
            
            for a in range(NUM_POINTS):
                if time.time() - start > time_limit:
                    break
                if a in S_without_i_set:
                    continue
                p_a = _all_points[a].reshape(1, 2)
                if len(S_without_i_arr) == 0:
                    S_with_a = p_a
                else:
                    S_with_a = np.vstack([S_without_i_arr, p_a])
                if not _check_valid(S_with_a):
                    continue
                for b in range(a + 1, NUM_POINTS):
                    if b in S_without_i_set:
                        continue
                    p_b = _all_points[b].reshape(1, 2)
                    S_with_ab = np.vstack([S_with_a, p_b])
                    if _check_valid(S_with_ab):
                        S_set = S_without_i_set | {a, b}
                        S_arr = S_with_ab
                        best_S = list(S_set)
                        best_score = len(S_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_S

def construct():
    """Construct a no-isosceles set of grid points in [0, N)^2."""
    start_time = time.time()
    
    # Initial greedy construction with natural order
    order = list(range(NUM_POINTS))
    S = _greedy_with_order(order)
    best_S = list(S)
    best_score = len(S)
    
    # Local search to improve
    if time.time() - start_time < DEADLINE - 10:
        improved_S = _local_search(best_S, time_limit=min(5.0, DEADLINE - time.time() - 5))
        if len(improved_S) > best_score:
            best_S = improved_S
            best_score = len(improved_S)
    
    # Try randomized restarts
    while time.time() - start_time < DEADLINE - 15:
        # Random order
        random.seed()
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        S_rand = _greedy_with_order(order)
        if len(S_rand) > best_score:
            best_S = S_rand
            best_score = len(S_rand)
        
        # Local search on best
        if time.time() - start_time < DEADLINE - 10:
            improved_S = _local_search(best_S, time_limit=min(3.0, DEADLINE - time.time() - 5))
            if len(improved_S) > best_score:
                best_S = improved_S
                best_score = len(improved_S)
        
        # If we've made good progress, keep trying
        if best_score >= 15:
            break
    
    # Convert indices back to (x, y) tuples
    result = [tuple(_all_points[i]) for i in best_S]
    return result
