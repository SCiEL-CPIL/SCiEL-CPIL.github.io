import time
import random
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute all isosceles triples as (i, j, k) where i < j < k
# A triple is isosceles if any two of the three pairwise distances are equal
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles triple list
_isosceles_triples = []
for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        for k in range(j + 1, N_TOTAL):
            di_k = _D2[i, k]
            dj_k = _D2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                _isosceles_triples.append((i, j, k))

# For each point p, precompute which triples contain p
# triple_index[p] = list of triple indices containing p
_triple_index = [[] for _ in range(N_TOTAL)]
for t_idx, (i, j, k) in enumerate(_isosceles_triples):
    _triple_index[i].append(t_idx)
    _triple_index[j].append(t_idx)
    _triple_index[k].append(t_idx)

# For each pair (i, j), precompute bitmask of points k that form isosceles with i,j
# Using Python ints (256 bits)
iso_c_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]
for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        mask = 0
        for k in range(N_TOTAL):
            if k == i or k == j:
                continue
            di_k = _D2[i, k]
            dj_k = _D2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                mask |= (1 << k)
        iso_c_mask[i][j] = mask
        iso_c_mask[j][i] = mask

# Store as numpy array of 4 uint64 per pair
iso_c_arr = np.zeros((N_TOTAL, N_TOTAL, 4), dtype=np.uint64)
for i in range(N_TOTAL):
    for j in range(N_TOTAL):
        m = iso_c_mask[i][j]
        iso_c_arr[i, j, 0] = m & 0xFFFFFFFFFFFFFFFF
        iso_c_arr[i, j, 1] = (m >> 64) & 0xFFFFFFFFFFFFFFFF
        iso_c_arr[i, j, 2] = (m >> 128) & 0xFFFFFFFFFFFFFFFF
        iso_c_arr[i, j, 3] = (m >> 192) & 0xFFFFFFFFFFFFFFFF


def _mask_to_arr(m):
    return np.array([m & 0xFFFFFFFFFFFFFFFF,
                     (m >> 64) & 0xFFFFFFFFFFFFFFFF,
                     (m >> 128) & 0xFFFFFFFFFFFFFFFF,
                     (m >> 192) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)


def _can_add(p, S_mask, S_indices):
    """Can point p be added to set S without creating an isosceles triple?"""
    if len(S_indices) < 2:
        return True
    S_mask_arr = _mask_to_arr(S_mask)
    masks = iso_c_arr[p, S_indices, :]
    return not np.any(masks & S_mask_arr)


def _is_blocked(p, S_mask, S_indices):
    """Is point p blocked by set S? (adding p would create isosceles)"""
    if len(S_indices) < 2:
        return False
    S_mask_arr = _mask_to_arr(S_mask)
    masks = iso_c_arr[S_indices, p, :]
    return bool(np.any(masks & S_mask_arr))


def _is_no_isosceles(S_mask, S_indices):
    """Check if set S contains no isosceles triple."""
    n = len(S_indices)
    if n < 3:
        return True
    S_mask_arr = _mask_to_arr(S_mask)
    for i in range(n):
        for j in range(i + 1, n):
            if np.any(iso_c_arr[S_indices[i], S_indices[j], :] & S_mask_arr):
                return False
    return True


def _is_maximal(S_mask, S_indices):
    """Check if S is maximal: every point outside S is blocked."""
    S_set = set(S_indices)
    for p in range(N_TOTAL):
        if p not in S_set and not _is_blocked(p, S_mask, S_indices):
            return False
    return True


def _greedy_set_cover(deadline_margin=15, rng=None):
    """
    Greedy set cover: select points to 'hit' all isosceles triples.
    At each step, pick the point that hits the most uncovered triples.
    This gives a hitting set, which is a no-isosceles maximal set.
    """
    if rng is None:
        rng = random.Random()
    
    # Track which triples are still uncovered
    n_triples = len(_isosceles_triples)
    uncovered = np.ones(n_triples, dtype=np.int32)  # 1 = uncovered, 0 = covered
    
    S_indices = []
    S_mask = 0
    S_set = set()
    
    while True:
        if time.time() > DEADLINE - deadline_margin:
            return None
        
        # Check if all triples are covered
        if uncovered.sum() == 0:
            break
        
        # For each candidate point not in S, count how many uncovered triples it hits
        best_p = None
        best_hits = 0
        
        for p in range(N_TOTAL):
            if p in S_set:
                continue
            
            # Count uncovered triples containing p
            hits = 0
            for t_idx in _triple_index[p]:
                if uncovered[t_idx]:
                    hits += 1
            
            if hits > best_hits:
                best_hits = hits
                best_p = p
        
        if best_p is None or best_hits == 0:
            # No point can hit any remaining triple - shouldn't happen if triples exist
            # But if we have a valid no-isosceles set, all triples should be hit
            break
        
        # Add best_p to S
        S_indices.append(best_p)
        S_mask |= (1 << best_p)
        S_set.add(best_p)
        
        # Mark all triples containing best_p as covered
        for t_idx in _triple_index[best_p]:
            uncovered[t_idx] = 0
    
    # Now S is a hitting set (no isosceles triple is fully contained in S)
    # But we need S to be MAXIMAL - every point outside S must be blocked
    # Extend S greedily to make it maximal while maintaining no-isosceles
    changed = True
    while changed:
        if time.time() > DEADLINE - deadline_margin:
            return None
        changed = False
        for p in range(N_TOTAL):
            if p in S_set:
                continue
            if not _is_blocked(p, S_mask, S_indices):
                if _can_add(p, S_mask, S_indices):
                    S_indices.append(p)
                    S_mask |= (1 << p)
                    S_set.add(p)
                    changed = True
    
    return S_indices


def _greedy_maximal_extend(deadline_margin=15, rng=None):
    """
    Greedy maximal extension: build a maximal no-isosceles set by
    at each step picking the addable point that blocks the most unblocked points.
    """
    if rng is None:
        rng = random.Random()
    
    S_indices = []
    S_mask = 0
    S_set = set()
    
    while True:
        if time.time() > DEADLINE - deadline_margin:
            return None
        
        # Find all addable points
        addable = []
        for p in range(N_TOTAL):
            if p not in S_set and _can_add(p, S_mask, S_indices):
                addable.append(p)
        
        if not addable:
            break
        
        # Check if already maximal
        if _is_maximal(S_mask, S_indices):
            break
        
        # Find unblocked points outside S
        unblocked = []
        for p in range(N_TOTAL):
            if p not in S_set and not _is_blocked(p, S_mask, S_indices):
                unblocked.append(p)
        
        if not unblocked:
            break
        
        # For each addable point, count how many unblocked points it would block
        best_p = None
        best_gain = -1
        S_mask_arr = _mask_to_arr(S_mask)
        
        for p in addable:
            gain = 1  # blocks itself
            for q in unblocked:
                if q == p:
                    continue
                # q would be blocked by S ∪ {p} iff exists a in S: (q, p, a) isosceles
                # i.e., a in iso_c_mask[q][p]
                if np.any(iso_c_arr[q, p, :] & S_mask_arr):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        S_indices.append(best_p)
        S_mask |= (1 << best_p)
        S_set.add(best_p)
    
    return S_indices


def _reduce_to_minimal(S_indices, deadline_margin=10):
    """
    Iteratively try to remove points from S while maintaining maximal no-isosceles.
    Use randomized order for diversity.
    """
    S = list(S_indices)
    changed = True
    rng = random.Random()
    
    while changed:
        if time.time() > DEADLINE - deadline_margin:
            break
        changed = False
        
        # Try removing in random order
        order = list(range(len(S)))
        rng.shuffle(order)
        
        for i in order:
            if len(S) - 1 < 2:
                continue
            
            S_test = S[:i] + S[i+1:]
            S_mask_test = 0
            for x in S_test:
                S_mask_test |= (1 << x)
            
            if _is_no_isosceles(S_mask_test, S_test) and _is_maximal(S_mask_test, S_test):
                S = S_test
                changed = True
                break
    
    return S


def _two_opt_reduce(S_indices, deadline_margin=5):
    """
    2-opt reduction: try removing one point and adding a different one.
    This can find smaller sets than simple removal.
    """
    S = list(S_indices)
    best_size = len(S)
    improved = True
    rng = random.Random()
    
    while improved:
        if time.time() > DEADLINE - deadline_margin:
            break
        improved = False
        
        n_S = len(S)
        # Try removing each point
        for i in range(n_S):
            if time.time() > DEADLINE - deadline_margin:
                break
            
            S_without = S[:i] + S[i+1:]
            S_mask_without = 0
            for x in S_without:
                S_mask_without |= (1 << x)
            
            if not _is_no_isosceles(S_mask_without, S_without):
                continue
            
            # Try adding each point not in S_without
            S_mask_without_arr = _mask_to_arr(S_mask_without)
            S_without_set = set(S_without)
            
            for p_add in range(N_TOTAL):
                if p_add in S_without_set:
                    continue
                if not _can_add(p_add, S_mask_without, S_without):
                    continue
                
                S_new = S_without + [p_add]
                S_mask_new = S_mask_without | (1 << p_add)
                
                if not _is_no_isosceles(S_mask_new, S_new):
                    continue
                if not _is_maximal(S_mask_new, S_new):
                    continue
                
                # Found a valid replacement - reduce it
                reduced = _reduce_to_minimal(S_new, deadline_margin=3)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    S = reduced
                    improved = True
                    break
            
            if improved:
                break
    
    return S


def _random_core_greedy(core_size, deadline_margin=10, rng=None):
    """
    Start with a random core, then greedily extend to maximal.
    """
    if rng is None:
        rng = random.Random()
    
    # Pick random core
    core = rng.sample(range(N_TOTAL), core_size)
    core_mask = 0
    for x in core:
        core_mask |= (1 << x)
    
    if not _is_no_isosceles(core_mask, core):
        return None
    
    S_indices = list(core)
    S_mask = core_mask
    S_set = set(core)
    
    # Greedy extend
    while True:
        if time.time() > DEADLINE - deadline_margin:
            return None
        
        addable = []
        for p in range(N_TOTAL):
            if p not in S_set and _can_add(p, S_mask, S_indices):
                addable.append(p)
        
        if not addable:
            break
        
        if _is_maximal(S_mask, S_indices):
            break
        
        unblocked = []
        for p in range(N_TOTAL):
            if p not in S_set and not _is_blocked(p, S_mask, S_indices):
                unblocked.append(p)
        
        if not unblocked:
            break
        
        best_p = None
        best_gain = -1
        S_mask_arr = _mask_to_arr(S_mask)
        
        for p in addable:
            gain = 1
            for q in unblocked:
                if q == p:
                    continue
                if np.any(iso_c_arr[q, p, :] & S_mask_arr):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        S_indices.append(best_p)
        S_mask |= (1 << best_p)
        S_set.add(best_p)
    
    return S_indices


def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy set cover (hitting set approach)
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        result = _greedy_set_cover(deadline_margin=20, rng=rng)
        if result is None:
            continue
        
        reduced = _reduce_to_minimal(result, deadline_margin=10)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 8:
            return [_ALL_POINTS[i] for i in best_set]
    
    # Strategy 2: Greedy maximal extension with different random orders
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 50:
            break
        
        result = _greedy_maximal_extend(deadline_margin=15, rng=rng)
        if result is None:
            continue
        
        reduced = _reduce_to_minimal(result, deadline_margin=8)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 8:
            return [_ALL_POINTS[i] for i in best_set]
    
    # Strategy 3: Random core + greedy extension
    for attempt in range(80):
        if time.time() - start_time > DEADLINE - 40:
            break
        
        core_size = rng.choice([2, 3, 4, 5])
        result = _random_core_greedy(core_size, deadline_margin=10, rng=rng)
        if result is None:
            continue
        
        reduced = _reduce_to_minimal(result, deadline_margin=8)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 8:
            return [_ALL_POINTS[i] for i in best_set]
    
    # Strategy 4: 2-opt reduction on best found
    if best_set is not None and best_size > 8:
        improved = _two_opt_reduce(best_set, deadline_margin=15)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Strategy 5: More 2-opt iterations if time allows
    if best_set is not None and best_size > 8:
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            improved = _two_opt_reduce(best_set, deadline_margin=5)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            else:
                break
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = []
        chosen_mask = 0
        chosen_set = set()
        for i in range(N_TOTAL):
            if i not in chosen_set and _can_add(i, chosen_mask, chosen):
                chosen.append(i)
                chosen_mask |= (1 << i)
                chosen_set.add(i)
        best_set = _reduce_to_minimal(chosen, deadline_margin=5)
    
    return [_ALL_POINTS[i] for i in best_set]
