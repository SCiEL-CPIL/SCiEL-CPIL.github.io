import itertools
import time
import random
import math
import numpy as np

N = 16
V = N * N

def _precompute():
    """Precompute distance matrix and conflict masks."""
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Compute squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    
    # Precompute conflict masks
    C = [[0] * V for _ in range(V)]
    
    # For each pair (i,j), find all k that form an isosceles triangle with i,j
    # Condition: d(i,j)==d(i,k) OR d(i,j)==d(j,k) OR d(i,k)==d(j,k)
    
    # Precompute for each point i, the set of points at each distance
    dist_groups = [{} for _ in range(V)]
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            d = int(D[i, j])
            if d not in dist_groups[i]:
                dist_groups[i][d] = 0
            dist_groups[i][d] |= (1 << j)
    
    # Precompute perpendicular bisector masks: for each pair (i,j), 
    # mask of k where D[i,k] == D[j,k]
    bisector = [[0] * V for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            mask = 0
            for k in range(V):
                if k != i and k != j:
                    if D[i, k] == D[j, k]:
                        mask |= (1 << k)
            bisector[i][j] = mask
            bisector[j][i] = mask
    
    # Now compute C[i][j] = mask of k that conflict with pair (i,j)
    for i in range(V):
        for j in range(i + 1, V):
            dij = int(D[i, j])
            mask = 0
            # k where d(i,k) == d(i,j)
            mask |= dist_groups[i].get(dij, 0)
            # k where d(j,k) == d(i,j)
            mask |= dist_groups[j].get(dij, 0)
            # k where d(i,k) == d(j,k)
            mask |= bisector[i][j]
            # Remove i and j themselves
            mask &= ~(1 << i)
            mask &= ~(1 << j)
            C[i][j] = mask
            C[j][i] = mask
    
    # Precompute for each point p, the union of C[p][q] for all q
    # This helps with incremental blocked computation
    point_conflict_union = [0] * V
    for p in range(V):
        union = 0
        for q in range(V):
            if q != p:
                union |= C[p][q]
        point_conflict_union[p] = union
    
    return points, D, C, bisector, dist_groups, point_conflict_union

_points, _D, _C, _bisector, _dist_groups, _point_conflict_union = None, None, None, None, None, None

def _init():
    global _points, _D, _C, _bisector, _dist_groups, _point_conflict_union
    if _points is None:
        _points, _D, _C, _bisector, _dist_groups, _point_conflict_union = _precompute()

def _get_blocked_mask(S_mask, C):
    """Calculate the union of conflict masks for all pairs in S."""
    blocked = 0
    indices = []
    m = S_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
        
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            blocked |= C[indices[i]][indices[j]]
            
    return blocked

def _is_valid(S_mask, C):
    """Check if S is a valid no-isosceles set."""
    indices = []
    m = S_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
        
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if S_mask & C[indices[i]][indices[j]]:
                return False
    return True

def _is_maximal(S_mask, C, V):
    """Check if S is maximal."""
    blocked = _get_blocked_mask(S_mask, C)
    return (blocked | S_mask) == (1 << V) - 1

def _greedy_build(seed, C, V, point_conflict_union):
    """Greedy construction with incremental blocked tracking."""
    rng = random.Random(seed)
    S_mask = 0
    blocked = 0
    U_mask = (1 << V) - 1
    
    while True:
        candidates_mask = U_mask & ~blocked & ~S_mask
        
        if not candidates_mask:
            break
            
        # Find candidates
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            m ^= lsb
            
        if not candidates:
            break
            
        # Get current indices for scoring
        curr_indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            curr_indices.append(idx)
            m ^= lsb
            
        # Score candidates: number of new points blocked
        best_score = -1
        best_candidate = None
        
        rng.shuffle(candidates)
        
        for p in candidates:
            # Check validity: p must not be in conflict with any pair in S
            valid = True
            for idx in curr_indices:
                if S_mask & C[p][idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new blocked incrementally
            # new_blocked = blocked | (union of C[p][q] for q in S)
            # But we need to be careful: C[p][q] may include points already blocked
            new_blocked = blocked
            for idx in curr_indices:
                new_blocked |= C[p][idx]
            
            # Score: number of newly blocked points (not in S, not already blocked)
            new_blocked_count = bin(new_blocked & ~blocked & ~S_mask).count('1')
            
            # Secondary score: prefer points that are "central" (block more in future)
            # Approximate by counting how many other candidates would be blocked
            future_block = 0
            for q in candidates:
                if q != p and (new_blocked & (1 << q)):
                    future_block += 1
            
            score = new_blocked_count * 2 + future_block
            
            if score > best_score:
                best_score = score
                best_candidate = p
            elif score == best_score and best_candidate is not None:
                if rng.random() < 0.5:
                    best_candidate = p
                    
        if best_candidate is None:
            break
            
        # Update
        S_mask |= (1 << best_candidate)
        U_mask &= ~(1 << best_candidate)
        # Update blocked
        for idx in curr_indices:
            blocked |= C[best_candidate][idx]
        
    return S_mask

def _local_search(S_mask, C, V, max_iters=50, time_limit=None):
    """Local search to minimize size with simulated annealing."""
    start_time = time.time() if time_limit else None
    
    for _ in range(max_iters):
        if time_limit and time.time() - start_time > time_limit:
            break
            
        improved = False
        indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        
        # Try removing points (multiple passes)
        for _pass in range(3):
            for idx in list(indices):
                new_mask = S_mask & ~(1 << idx)
                if _is_maximal(new_mask, C, V):
                    S_mask = new_mask
                    improved = True
                    # Rebuild indices
                    indices = []
                    m = S_mask
                    while m:
                        lsb = m & -m
                        idx2 = lsb.bit_length() - 1
                        indices.append(idx2)
                        m ^= lsb
                    break
            
            if improved:
                break
        
        if improved:
            continue
        
        # Try swaps with scoring
        best_swap_score = float('inf')
        best_p, best_q = None, None
        
        for p in indices:
            base_mask = S_mask & ~(1 << p)
            
            # Candidates for q: not in S, and not in base_mask's blocked
            blocked_base = _get_blocked_mask(base_mask, C)
            q_candidates = ((1 << V) - 1) & ~S_mask & ~blocked_base
            
            m = q_candidates
            while m:
                lsb = m & -m
                q = lsb.bit_length() - 1
                m ^= lsb
                
                trial_mask = base_mask | (1 << q)
                
                # Check validity
                if not _is_valid(trial_mask, C):
                    continue
                    
                # Check maximality
                if not _is_maximal(trial_mask, C, V):
                    continue
                
                # Score: prefer swaps that block more points (more "efficient" coverage)
                blocked_trial = _get_blocked_mask(trial_mask, C)
                score = -bin(blocked_trial).count('1')
                
                if score < best_swap_score:
                    best_swap_score = score
                    best_p, best_q = p, q
        
        if best_p is not None:
            S_mask = (S_mask & ~(1 << best_p)) | (1 << best_q)
            improved = True
        
        if not improved:
            break
            
    return S_mask

def construct():
    _init()
    C = _C
    V = len(_points)
    point_conflict_union = _point_conflict_union
    
    best_size = float('inf')
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 280.0
    
    trial_count = 0
    
    while time.time() < deadline:
        seed = trial_count * 7919 + 12345
        
        S_mask = _greedy_build(seed, C, V, point_conflict_union)
        
        # Local search with time limit
        remaining = deadline - time.time()
        ls_time = min(30.0, remaining * 0.3)
        S_mask = _local_search(S_mask, C, V, max_iters=50, time_limit=ls_time)
        
        size = bin(S_mask).count('1')
        
        if size < best_size:
            best_size = size
            best_mask = S_mask
            if best_size <= 5:
                break
                
        trial_count += 1
    
    # Final polish: try more aggressive local search on best
    remaining = deadline - time.time()
    if remaining > 10:
        S_mask = _local_search(best_mask, C, V, max_iters=200, time_limit=remaining - 5)
        size = bin(S_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = S_mask
    
    # Convert mask to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(_points[idx])
        m ^= lsb
    
    return result
