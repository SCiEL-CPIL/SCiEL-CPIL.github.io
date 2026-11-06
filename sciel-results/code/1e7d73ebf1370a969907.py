"""
Minimum maximal no-isosceles set on [0, N)^2.
Combined strategy:
  1. Coverage-based greedy construction (prioritize points that block most).
  2. Fast vectorized feasibility and maximality checks.
  3. Aggressive minimality reduction.
  4. Local search with 1-swap moves.
  5. Multiple random restarts with time-budgeted loops.
"""
import time
import random
import numpy as np

N = 16
DEADLINE = 315.0  # safety margin before 330s kill

# Precompute all grid points
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
N_TOTAL = len(ALL_POINTS)

# Precompute pairwise squared distances as a matrix
_POINTS_ARR = np.array(ALL_POINTS, dtype=np.int32)
_px = _POINTS_ARR[:, 0]
_py = _POINTS_ARR[:, 1]
_dx = _px[:, None] - _px[None, :]
_dy = _py[:, None] - _py[None, :]
DIST2 = _dx * _dx + _dy * _dy  # shape (256, 256)


def _is_maximal_vec(mask):
    """
    mask: boolean array of shape (N_TOTAL,) indicating which points are in S.
    Returns True if S is maximal (no point outside S can be added without creating isosceles).
    Assumes S has at least 2 points and is no-isosceles.
    """
    S_idx = np.where(mask)[0]
    n_S = len(S_idx)
    if n_S < 2:
        return False
    
    non_S_mask = ~mask
    non_S_idx = np.where(non_S_mask)[0]
    
    if len(non_S_idx) == 0:
        return True
    
    # dist_p_S: shape (n_nonS, n_S)
    dist_p_S = DIST2[non_S_idx[:, None], S_idx[None, :]]
    dist_S_S = DIST2[S_idx[:, None], S_idx[None, :]]
    
    blocked = np.zeros(len(non_S_idx), dtype=bool)
    
    # Check condition 1: two distances from p to S are equal
    for p_idx in range(len(non_S_idx)):
        if blocked[p_idx]:
            continue
        row = dist_p_S[p_idx]
        if len(row) < 2:
            continue
        seen = set()
        for v in row:
            if v in seen:
                blocked[p_idx] = True
                break
            seen.add(v)
    
    # Check condition 2: dist_p_S[p,i] == dist_S_S[i,j] for some i != j
    S_dist_sets = []
    for i in range(n_S):
        row = dist_S_S[i]
        row = np.delete(row, i)
        S_dist_sets.append(set(row.tolist()))
    
    for p_idx in range(len(non_S_idx)):
        if blocked[p_idx]:
            continue
        for i in range(n_S):
            d = dist_p_S[p_idx, i]
            if d in S_dist_sets[i]:
                blocked[p_idx] = True
                break
    
    return bool(np.all(blocked))


def _can_add_vec(p_idx, S_idx):
    """
    Check if point p_idx can be added to S (given by indices S_idx) without creating isosceles.
    """
    if len(S_idx) < 2:
        return True
    
    dist_p_S = DIST2[p_idx, S_idx]
    dist_S_S = DIST2[S_idx[:, None], S_idx[None, :]]
    
    n_S = len(S_idx)
    for i in range(n_S):
        for j in range(i + 1, n_S):
            if dist_p_S[i] == dist_p_S[j]:
                return False
            if dist_p_S[i] == dist_S_S[i, j]:
                return False
            if dist_p_S[j] == dist_S_S[i, j]:
                return False
    return True


def _reduce_to_minimal(S_idx):
    """
    Iteratively remove points from S if the remaining set is still maximal.
    """
    S_list = list(S_idx)
    changed = True
    while changed:
        changed = False
        for i in range(len(S_list)):
            p = S_list[i]
            S_test = S_list[:i] + S_list[i+1:]
            
            if len(S_test) < 2:
                continue
            
            mask = np.zeros(N_TOTAL, dtype=bool)
            for idx in S_test:
                mask[idx] = True
            
            if _is_maximal_vec(mask):
                S_list = S_test
                changed = True
                break
                
    return S_list


def _coverage_greedy(rng, start_mode="random"):
    """
    Greedy construction using coverage heuristic.
    At each step, pick the point that blocks the most currently-unblocked points.
    start_mode: 'random', 'corner', 'center'
    """
    mask = np.zeros(N_TOTAL, dtype=bool)
    S_list = []
    
    # Pick starting point based on mode
    if start_mode == "random":
        start = rng.randint(0, N_TOTAL - 1)
    elif start_mode == "corner":
        # Pick a random corner
        corners = [0, (N-1)*N, (N-1)*N + (N-1), (N-1)]
        start = rng.choice(corners)
    elif start_mode == "center":
        cx, cy = (N-1)//2, (N-1)//2
        start = cy * N + cx
    else:
        start = rng.randint(0, N_TOTAL - 1)
    
    mask[start] = True
    S_list.append(start)
    
    while True:
        # Find all candidates that can be added
        candidates = []
        for p in range(N_TOTAL):
            if mask[p]:
                continue
            if _can_add_vec(p, np.array(S_list, dtype=np.int32)):
                candidates.append(p)
        
        if not candidates:
            break
        
        # Score each candidate by how many new points it blocks
        best_p = None
        best_score = -1
        
        # For efficiency, sample if too many candidates
        if len(candidates) > 80:
            candidates = rng.sample(candidates, 80)
        
        S_arr = np.array(S_list, dtype=np.int32)
        
        for p in candidates:
            # Count how many currently-unblocked points would become blocked
            score = 0
            for q in range(N_TOTAL):
                if mask[q] or q == p:
                    continue
                # Check if q is currently unblocked
                already_blocked = False
                for i in range(len(S_list)):
                    a = S_list[i]
                    dq_a = DIST2[q, a]
                    for j in range(i + 1, len(S_list)):
                        b = S_list[j]
                        dq_b = DIST2[q, b]
                        d_ab = DIST2[a, b]
                        if dq_a == dq_b or dq_a == d_ab or dq_b == d_ab:
                            already_blocked = True
                            break
                    if already_blocked:
                        break
                if already_blocked:
                    continue
                
                # Check if q would be blocked by adding p
                dp_q = DIST2[p, q]
                for a in S_list:
                    dp_a = DIST2[p, a]
                    dq_a = DIST2[q, a]
                    if dp_a == dq_a or dp_a == dp_q or dq_a == dp_q:
                        score += 1
                        break
        
        if best_score < score and score > 0:
            best_score = score
            best_p = p
        
        if best_p is None:
            break
        
        mask[best_p] = True
        S_list.append(best_p)
    
    return S_list


def _local_search(S_list, rng):
    """
    Try 1-swap with random perturbation: remove one point, add another, check maximality, reduce.
    """
    S_set = set(S_list)
    n_S = len(S_list)
    
    # Shuffle order for randomness
    order = list(range(n_S))
    rng.shuffle(order)
    
    for i in order:
        p_remove = S_list[i]
        S_without = S_list[:i] + S_list[i+1:]
        S_without_set = set(S_without)
        
        if len(S_without) < 2:
            continue
        
        # Try adding each point not in S_without
        for p_add in range(N_TOTAL):
            if p_add in S_without_set:
                continue
            
            if not _can_add_vec(p_add, np.array(S_without, dtype=np.int32)):
                continue
            
            S_new = S_without + [p_add]
            
            # Check maximality
            mask = np.zeros(N_TOTAL, dtype=bool)
            for idx in S_new:
                mask[idx] = True
            
            if _is_maximal_vec(mask):
                reduced = _reduce_to_minimal(S_new)
                if len(reduced) < n_S:
                    return reduced
                    
    return None


def _perturb_and_repair(S_list, rng):
    """
    Randomly perturb S by removing 1-2 points and re-adding to try to escape local minima.
    """
    if len(S_list) < 4:
        return None
    
    S_test = list(S_list)
    # Remove 1-2 random points
    n_remove = rng.randint(1, min(2, len(S_test)-2))
    to_remove = rng.sample(range(len(S_test)), n_remove)
    to_remove.sort(reverse=True)
    for idx in to_remove:
        S_test.pop(idx)
    
    # Try to extend back to maximal using coverage greedy
    S_set = set(S_test)
    mask = np.zeros(N_TOTAL, dtype=bool)
    for p in S_test:
        mask[p] = True
    
    # Continue with coverage greedy from current state
    while True:
        candidates = []
        for p in range(N_TOTAL):
            if mask[p]:
                continue
            if _can_add_vec(p, np.array(S_test, dtype=np.int32)):
                candidates.append(p)
        
        if not candidates:
            break
        
        if len(candidates) > 80:
            candidates = rng.sample(candidates, 80)
        
        best_p = None
        best_score = -1
        
        for p in candidates:
            score = 0
            for q in range(N_TOTAL):
                if mask[q] or q == p:
                    continue
                already_blocked = False
                for i in range(len(S_test)):
                    a = S_test[i]
                    dq_a = DIST2[q, a]
                    for j in range(i + 1, len(S_test)):
                        b = S_test[j]
                        dq_b = DIST2[q, b]
                        d_ab = DIST2[a, b]
                        if dq_a == dq_b or dq_a == d_ab or dq_b == d_ab:
                            already_blocked = True
                            break
                    if already_blocked:
                        break
                if already_blocked:
                    continue
                
                dp_q = DIST2[p, q]
                for a in S_test:
                    dp_a = DIST2[p, a]
                    dq_a = DIST2[q, a]
                    if dp_a == dq_a or dp_a == dp_q or dq_a == dp_q:
                        score += 1
                        break
            
            if best_score < score and score > 0:
                best_score = score
                best_p = p
        
        if best_p is None:
            break
        
        mask[best_p] = True
        S_test.append(best_p)
    
    reduced = _reduce_to_minimal(S_test)
    return reduced


def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Multiple diverse coverage greedy + reduction
    for attempt in range(150):
        if time.time() - start_time > DEADLINE:
            break
        
        # Cycle through different starting modes
        if attempt % 3 == 0:
            chosen = _coverage_greedy(rng, "random")
        elif attempt % 3 == 1:
            chosen = _coverage_greedy(rng, "corner")
        else:
            chosen = _coverage_greedy(rng, "center")
        
        if not chosen:
            continue
        
        # Extend to maximal if needed
        S_set = set(chosen)
        for p in range(N_TOTAL):
            if p in S_set:
                continue
            if _can_add_vec(p, np.array(chosen, dtype=np.int32)):
                chosen.append(p)
                S_set.add(p)
        
        reduced = _reduce_to_minimal(chosen)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 6:
            break
    
    # Strategy 2: Local search on best set
    if best_set is not None:
        for attempt in range(80):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = _local_search(best_set, rng)
            if improved is not None:
                best_set = improved
                best_size = len(best_set)
                
            if best_size <= 6:
                break
    
    # Strategy 3: Perturbation and repair
    if best_set is not None:
        for attempt in range(40):
            if time.time() - start_time > DEADLINE:
                break
            
            perturbed = _perturb_and_repair(best_set, rng)
            if perturbed is not None and len(perturbed) < best_size:
                best_set = perturbed
                best_size = len(best_set)
            
            if best_size <= 6:
                break
    
    if best_set is None:
        # Fallback
        chosen = _coverage_greedy(rng, "random")
        best_set = _reduce_to_minimal(chosen)
        
    # Convert back to (x, y) tuples
    return [ALL_POINTS[i] for i in best_set]
