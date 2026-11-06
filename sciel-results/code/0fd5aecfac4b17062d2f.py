import time
import random
import itertools
import numpy as np
from collections import defaultdict

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute isosceles conflicts
# conflict[i][j] = set of points k such that {i, j, k} forms an isosceles triangle
# When i and j are in S, no k in conflict[i][j] can be added to S
_conflict = None
_dist_matrix = None

def _precompute():
    global _conflict, _dist_matrix
    if _conflict is not None:
        return
    
    V = NUM_POINTS
    pts = np.array(POINTS, dtype=np.int32)
    
    # Compute squared distance matrix
    _dist_matrix = np.zeros((V, V), dtype=np.int32)
    for i in range(V):
        diff = pts[i] - pts
        _dist_matrix[i] = np.sum(diff**2, axis=1)
    
    # For each pair (i, j), compute conflict set
    _conflict = [set() for _ in range(V)]
    conflict_pairs = defaultdict(set)  # (i,j) -> set of k
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = _dist_matrix[i, j]
            # Find all k where {i,j,k} is isosceles
            # Isosceles if: d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k)
            mask = np.zeros(V, dtype=bool)
            mask[i] = False
            mask[j] = False
            
            # d(i,j) == d(i,k)
            mask |= (_dist_matrix[i] == dij)
            # d(i,j) == d(j,k)  
            mask |= (_dist_matrix[j] == dij)
            # d(i,k) == d(j,k)
            mask |= (_dist_matrix[i] == _dist_matrix[j])
            
            mask[i] = False
            mask[j] = False
            
            ks = np.where(mask)[0]
            for k in ks:
                k = int(k)
                _conflict[i].add((j, k))
                _conflict[j].add((i, k))
                _conflict[k].add((i, j))

def _is_safe(p_idx, S_set):
    """Check if adding p_idx to S_set maintains no-isosceles property."""
    for j, k in _conflict[p_idx]:
        if j in S_set and k in S_set:
            return False
    return True

def _get_blocked(S_set):
    """Get set of points blocked by S_set (cannot be added)."""
    blocked = set()
    S_list = list(S_set)
    for i in range(len(S_list)):
        for j in range(i + 1, len(S_list)):
            a, b = S_list[i], S_list[j]
            # Points that form isosceles with a,b are blocked
            # We need to find all k such that {a,b,k} is isosceles
            # This is equivalent to: k is in conflict for pair (a,b)
            # But our _conflict is per-point. Let's compute directly.
            dij = _dist_matrix[a, b]
            # k where d(a,k)==dij or d(b,k)==dij or d(a,k)==d(b,k)
            for k in range(NUM_POINTS):
                if k == a or k == b:
                    continue
                dak = _dist_matrix[a, k]
                dbk = _dist_matrix[b, k]
                if dak == dij or dbk == dij or dak == dbk:
                    blocked.add(k)
    return blocked

def _is_maximal(S_set):
    """Check if S_set is maximal (no point can be added)."""
    S_set = set(S_set)
    blocked = _get_blocked(S_set)
    for p in range(NUM_POINTS):
        if p not in S_set and p not in blocked:
            return False
    return True

def _compute_blocking_power(p_idx, S_set, U_set):
    """Compute how many points in U_set would be blocked if p_idx is added to S_set."""
    S_trial = S_set | {p_idx}
    blocked_trial = _get_blocked(S_trial)
    # Points newly blocked = blocked_trial - _get_blocked(S_set)
    blocked_current = _get_blocked(S_set)
    newly_blocked = blocked_trial - blocked_current
    newly_blocked &= U_set
    newly_blocked.discard(p_idx)
    return len(newly_blocked)

def _branch_and_bound_construct(seed, time_limit=5.0):
    """
    Construct a maximal no-isosceles set using branch-and-bound with dynamic ordering.
    Prioritize points with high blocking power to minimize final set size.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    S = []
    S_set = set()
    U = set(range(NUM_POINTS))
    
    while U and time.time() - start_time < time_limit:
        # Compute blocking power for each candidate in U
        candidates = []
        for p in U:
            if not _is_safe(p, S_set):
                continue
            bp = _compute_blocking_power(p, S_set, U)
            candidates.append((bp, p))
        
        if not candidates:
            break
        
        # Sort by blocking power (descending) with random tie-breaking
        candidates.sort(key=lambda x: (-x[0], rng.random()))
        
        # Try top few candidates, pick the one that leads to smallest set
        best_choice = None
        best_size = float('inf')
        
        # Limit to top 3-5 for efficiency
        top_candidates = candidates[:min(5, len(candidates))]
        
        for bp, p in top_candidates:
            S_trial = S + [p]
            S_trial_set = S_set | {p}
            U_trial = U - {p}
            
            # Quick greedy completion to estimate final size
            S_greedy = list(S_trial)
            S_greedy_set = set(S_trial_set)
            U_greedy = set(U_trial)
            
            while U_greedy:
                # Pick point with max blocking power
                best_bp = -1
                best_p = None
                for q in U_greedy:
                    if not _is_safe(q, S_greedy_set):
                        continue
                    bp_q = _compute_blocking_power(q, S_greedy_set, U_greedy)
                    if bp_q > best_bp:
                        best_bp = bp_q
                        best_p = q
                if best_p is None:
                    break
                S_greedy.append(best_p)
                S_greedy_set.add(best_p)
                U_greedy.remove(best_p)
            
            if len(S_greedy) < best_size:
                best_size = len(S_greedy)
                best_choice = p
        
        if best_choice is None:
            break
        
        S.append(best_choice)
        S_set.add(best_choice)
        U.remove(best_choice)
        
        # Remove points that are now blocked
        blocked = _get_blocked(S_set)
        U -= blocked
    
    return S

def _local_search_minimize(S, max_iters=20, seed=0):
    """Try to minimize set size by removing points if maximality is maintained."""
    rng = random.Random(seed)
    S = list(S)
    S_set = set(S)
    
    for it in range(max_iters):
        improved = False
        # Try removing each point
        for idx in range(len(S)):
            trial = S[:idx] + S[idx+1:]
            trial_set = set(trial)
            
            if not trial:
                continue
            
            # Check validity
            valid = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    if not _is_safe(trial[i], trial_set - {trial[i]}):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check maximality
            if _is_maximal(trial_set):
                S = trial
                S_set = set(S)
                improved = True
                break
        
        if not improved:
            break
    
    return S

def _greedy_construct(rng):
    """Simple greedy construction for baseline."""
    S = []
    S_set = set()
    points = list(range(NUM_POINTS))
    rng.shuffle(points)
    
    for p in points:
        if _is_safe(p, S_set):
            S.append(p)
            S_set.add(p)
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    _precompute()
    
    best_set = None
    best_len = float('inf')
    
    # Phase 1: Branch-and-bound constructions with different seeds
    seed = 0
    while time.time() < deadline - 60:  # Leave time for local search
        S = _branch_and_bound_construct(seed, time_limit=3.0)
        
        if S and len(S) < best_len:
            best_len = len(S)
            best_set = S
        
        seed += 1
    
    # Phase 2: Local search refinement on best solution
    if best_set is not None and time.time() < deadline - 30:
        refined = _local_search_minimize(best_set, max_iters=50, seed=42)
        if len(refined) < best_len:
            best_len = len(refined)
            best_set = refined
    
    # Phase 3: More greedy restarts if time remains
    if time.time() < deadline - 10:
        rng = random.Random(1000)
        for _ in range(20):
            if time.time() >= deadline - 10:
                break
            S = _greedy_construct(rng)
            if S and len(S) < best_len:
                best_len = len(S)
                best_set = S
    
    if best_set is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_set = _greedy_construct(rng)
    
    return best_set
