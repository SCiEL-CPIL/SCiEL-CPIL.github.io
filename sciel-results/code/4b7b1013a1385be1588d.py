import itertools
import time
import random
import numpy as np

N = 16
V = N * N
FULL = (1 << V) - 1

def _precompute():
    """Precompute distance matrix and conflict masks."""
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    
    # Conflict mask C[i][j]: all k that form an isosceles triangle with pair (i,j)
    C = [[0] * V for _ in range(V)]
    for i in range(V):
        for j in range(i + 1, V):
            dij = int(D[i, j])
            mask = 0
            for k in range(V):
                if k == i or k == j:
                    continue
                dik = int(D[i, k])
                djk = int(D[j, k])
                if dij == dik or dij == djk or dik == djk:
                    mask |= (1 << k)
            C[i][j] = mask
            C[j][i] = mask
    return points, D, C

_points, _D, _C = None, None, None

def _init():
    global _points, _D, _C
    if _points is None:
        _points, _D, _C = _precompute()

def _get_indices(S_mask):
    indices = []
    m = S_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        indices.append(idx)
        m ^= lsb
    return indices

def _is_valid(S_mask, C):
    """Check no isosceles triangle in S."""
    indices = _get_indices(S_mask)
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if S_mask & C[indices[i]][indices[j]]:
                return False
    return True

def _get_blocked(S_mask, C):
    indices = _get_indices(S_mask)
    blocked = 0
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            blocked |= C[indices[i]][indices[j]]
    return blocked

def _is_maximal(S_mask, C):
    blocked = _get_blocked(S_mask, C)
    return (blocked | S_mask) == FULL

def _can_remove(S_mask, C, p):
    """Can we remove p and still be maximal?"""
    new_mask = S_mask & ~(1 << p)
    return _is_maximal(new_mask, C)

def _find_replacement(S_mask, C, p):
    """Find q not in S such that S - {p} + {q} is valid and maximal. Return q or -1."""
    base = S_mask & ~(1 << p)
    base_indices = _get_indices(base)
    base_blocked = _get_blocked(base, C)
    candidates = FULL & ~base & ~base_blocked
    m = candidates
    while m:
        lsb = m & -m
        q = lsb.bit_length() - 1
        m ^= lsb
        # Check validity: q must not conflict with any pair in base
        valid = True
        for idx in base_indices:
            if base & C[q][idx]:
                valid = False
                break
        if not valid:
            continue
        trial = base | (1 << q)
        if _is_maximal(trial, C):
            return q
    return -1

def _greedy_build(seed, C):
    """Greedy construction with incremental blocked tracking."""
    rng = random.Random(seed)
    S_mask = 0
    blocked = 0
    curr_indices = []
    
    while True:
        candidates_mask = FULL & ~blocked & ~S_mask
        if not candidates_mask:
            break
        
        candidates = _get_indices(candidates_mask)
        if not candidates:
            break
        
        rng.shuffle(candidates)
        
        best_score = -1
        best_candidate = None
        
        for p in candidates:
            # Check validity
            valid = True
            for idx in curr_indices:
                if S_mask & C[p][idx]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute new blocked
            new_blocked = blocked
            for idx in curr_indices:
                new_blocked |= C[p][idx]
            
            score = bin(new_blocked & ~blocked & ~S_mask).count('1')
            
            if score > best_score:
                best_score = score
                best_candidate = p
            elif score == best_score and best_candidate is not None:
                if rng.random() < 0.5:
                    best_candidate = p
        
        if best_candidate is None:
            break
        
        S_mask |= (1 << best_candidate)
        curr_indices.append(best_candidate)
        for idx in curr_indices[:-1]:
            blocked |= C[best_candidate][idx]
    
    return S_mask

def _local_search(S_mask, C, max_iters=200, deadline=None):
    """Local search: exact deletion + exact replacement passes."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        indices = _get_indices(S_mask)
        
        # Deletion pass
        for idx in indices:
            if _can_remove(S_mask, C, idx):
                S_mask &= ~(1 << idx)
                improved = True
                break
        if improved:
            continue
        
        # Replacement pass
        indices = _get_indices(S_mask)
        for idx in indices:
            q = _find_replacement(S_mask, C, idx)
            if q >= 0:
                S_mask = (S_mask & ~(1 << idx)) | (1 << q)
                improved = True
                break
        if improved:
            continue
        
        # No improvement
        break
    
    return S_mask

def _simulated_annealing(S_mask, C, start_temp, end_temp, n_steps, rng, deadline):
    """SA with move = remove a point (if still valid) + optionally add a point."""
    current = S_mask
    current_size = bin(current).count('1')
    best = current
    best_size = current_size
    temp = start_temp
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (step / max(1, n_steps - 1))
        
        indices = _get_indices(current)
        if not indices:
            break
        
        # Random move: remove a random point
        p = indices[rng.randint(0, len(indices) - 1)]
        trial = current & ~(1 << p)
        
        # Check validity
        valid = _is_valid(trial, C)
        if valid:
            new_size = bin(trial).count('1')
            delta = new_size - current_size  # negative = better
            if delta < 0 or rng.random() < np.exp(-delta / max(1e-9, temp)):
                current = trial
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best = current
                # Try to add a point to restore maximality (greedy)
                blocked = _get_blocked(current, C)
                candidates = FULL & ~current & ~blocked
                if candidates:
                    cand_list = _get_indices(candidates)
                    rng.shuffle(cand_list)
                    for q in cand_list:
                        trial2 = current | (1 << q)
                        if _is_valid(trial2, C):
                            current = trial2
                            current_size += 1
                            if current_size < best_size:
                                best_size = current_size
                                best = current
                            break
        else:
            # Try swap: remove p, add random q
            base = current & ~(1 << p)
            base_blocked = _get_blocked(base, C)
            candidates = FULL & ~base & ~base_blocked
            if candidates:
                cand_list = _get_indices(candidates)
                if cand_list:
                    q = cand_list[rng.randint(0, len(cand_list) - 1)]
                    trial = base | (1 << q)
                    if _is_valid(trial, C):
                        new_size = bin(trial).count('1')
                        delta = new_size - current_size
                        if delta < 0 or rng.random() < np.exp(-delta / max(1e-9, temp)):
                            current = trial
                            current_size = new_size
                            if current_size < best_size:
                                best_size = current_size
                                best = current
    
    return best, best_size

def construct():
    _init()
    C = _C
    
    best_size = float('inf')
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 290.0
    
    trial_count = 0
    rng_master = random.Random(42)
    
    # Phase 1: Greedy + local search
    while time.time() < deadline - 60.0:
        seed = trial_count * 7919 + 12345
        S_mask = _greedy_build(seed, C)
        S_mask = _local_search(S_mask, C, max_iters=100, deadline=deadline - 60.0)
        
        size = bin(S_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = S_mask
            if best_size <= 5:
                break
        
        trial_count += 1
    
    # Phase 2: SA on best solution
    if time.time() < deadline - 20.0:
        rng = random.Random(999)
        sa_best, sa_size = _simulated_annealing(
            best_mask, C,
            start_temp=5.0, end_temp=0.01,
            n_steps=5000,
            rng=rng,
            deadline=deadline - 20.0
        )
        if sa_size < best_size:
            best_size = sa_size
            best_mask = sa_best
    
    # Phase 3: Final exact polish
    if time.time() < deadline:
        S_mask = _local_search(best_mask, C, max_iters=200, deadline=deadline)
        size = bin(S_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = S_mask
    
    # Convert to list of points
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(_points[idx])
        m ^= lsb
    
    return result
