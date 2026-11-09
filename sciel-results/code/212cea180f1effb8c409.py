"""Optimized program: search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random
import numpy as np

_LCM = 232792560
_WEIGHTS = [0] + [_LCM // i for i in range(1, 23)]  # index by value 1..22


def _find_collision_fast(subset):
    """Find two different subsets with the same sum, or None if dissociated.
    Returns (mask1, mask2) where mask1 < mask2."""
    k = len(subset)
    if k == 0:
        return None
    scaled = np.array([_LCM // x for x in subset], dtype=np.int64)
    n = 1 << k
    sums = np.zeros(n, dtype=np.int64)
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    sorted_sums, sort_indices = np.sort(sums), np.argsort(sums)
    diffs = np.diff(sorted_sums)
    collision_pos = np.where(diffs == 0)[0]
    if len(collision_pos) > 0:
        pos = collision_pos[0]
        mask1 = int(sort_indices[pos])
        mask2 = int(sort_indices[pos + 1])
        return min(mask1, mask2), max(mask1, mask2)
    return None


def _min_pairwise_diff(scaled):
    """Compute the minimum absolute difference between any two distinct subset sums.
    Returns (min_diff, mask1, mask2) or (inf, -1, -1) if k==0."""
    k = len(scaled)
    if k == 0:
        return float('inf'), -1, -1
    n = 1 << k
    sums = np.zeros(n, dtype=np.int64)
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        prev = mask ^ lb
        sums[mask] = sums[prev] + scaled[idx]
    sorted_sums, sort_indices = np.sort(sums), np.argsort(sums)
    diffs = np.diff(sorted_sums)
    min_idx = int(np.argmin(diffs))
    min_diff = int(diffs[min_idx])
    mask1 = int(sort_indices[min_idx])
    mask2 = int(sort_indices[min_idx + 1])
    return min_diff, mask1, mask2


def _greedy_remove(subset, heuristic='largest', deadline=None):
    """Start with subset, iteratively remove elements causing collisions."""
    current = list(subset)
    while True:
        if deadline and time.time() > deadline:
            break
        col = _find_collision_fast(current)
        if col is None:
            break
        mask1, mask2 = col
        k = len(current)
        sym_diff = []
        for i in range(k):
            bit = 1 << i
            if (mask1 & bit) != (mask2 & bit):
                sym_diff.append(i)
        if heuristic == 'largest':
            to_remove = max(sym_diff, key=lambda i: current[i])
        elif heuristic == 'smallest':
            to_remove = min(sym_diff, key=lambda i: current[i])
        else:
            to_remove = random.choice(sym_diff)
        current.remove(current[to_remove])
    return current


def _try_add_elements(best, removed, deadline):
    """Try to add back removed elements one by one."""
    current = list(best)
    for x in removed:
        if time.time() > deadline:
            break
        candidate = current + [x]
        candidate.sort()
        if _find_collision_fast(candidate) is None:
            current = candidate
    return current


def _try_swaps(best, not_in_set, deadline):
    """Try swapping elements in best with elements not in best."""
    current = list(best)
    in_set = set(current)
    remaining = [x for x in not_in_set if x not in in_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(current):
            if time.time() > deadline:
                break
            for b in remaining:
                candidate = current[:]
                candidate[i] = b
                candidate.sort()
                if _find_collision_fast(candidate) is None:
                    current = candidate
                    in_set = set(current)
                    remaining = [x for x in not_in_set if x not in in_set]
                    improved = True
                    break
    return current


def _simulated_annealing(initial_set, deadline):
    """Randomized local search: swap elements in/out, accept moves that maintain dissociation
    and improve the minimum pairwise subset-sum difference (a smooth surrogate for robustness)."""
    current = list(initial_set)
    in_set = set(current)
    universe = list(range(1, 23))
    not_in_set = [x for x in universe if x not in in_set]
    
    best = list(current)
    best_len = len(current)
    best_min_diff = float('inf')
    if current:
        scaled = [_LCM // x for x in current]
        best_min_diff, _, _ = _min_pairwise_diff(scaled)
    
    start_temp = 1e6
    end_temp = 100.0
    total_time = max(deadline - time.time(), 1.0)
    elapsed = 0.0
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        elapsed = time.time() - (deadline - total_time)
        progress = min(elapsed / total_time, 1.0)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        if not current:
            # Try to add a random element
            if not_in_set:
                x = rng.choice(not_in_set)
                candidate = current + [x]
                candidate.sort()
                if _find_collision_fast(candidate) is None:
                    current = candidate
                    in_set = set(current)
                    not_in_set = [x for x in universe if x not in in_set]
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(current)
                        scaled = [_LCM // x for x in current]
                        best_min_diff, _, _ = _min_pairwise_diff(scaled)
            continue
        
        # Choose a random move type
        move_type = rng.random()
        
        if move_type < 0.5 and not_in_set:
            # Swap: remove one from current, add one from not_in_set
            i = rng.randrange(len(current))
            b = rng.choice(not_in_set)
            candidate = current[:]
            candidate.pop(i)
            candidate.append(b)
            candidate.sort()
            if _find_collision_fast(candidate) is None:
                # Compute min diff for candidate
                scaled_c = [_LCM // x for x in candidate]
                cand_min_diff, _, _ = _min_pairwise_diff(scaled_c)
                # Accept if same size and better min_diff, or larger size
                if len(candidate) > len(current):
                    accept = True
                elif cand_min_diff > best_min_diff or (cand_min_diff == best_min_diff and rng.random() < 0.5):
                    accept = True
                else:
                    # Metropolis: accept worse with probability exp(-delta/temp)
                    delta = best_min_diff - cand_min_diff
                    if delta <= 0:
                        accept = True
                    else:
                        accept = rng.random() < np.exp(-delta / max(temp, 1e-10))
                if accept:
                    current = candidate
                    in_set = set(current)
                    not_in_set = [x for x in universe if x not in in_set]
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(current)
                        best_min_diff, _, _ = _min_pairwise_diff(scaled_c)
                    elif cand_min_diff > best_min_diff:
                        best_min_diff = cand_min_diff
        
        elif move_type < 0.7 and not_in_set:
            # Add an element
            b = rng.choice(not_in_set)
            candidate = current + [b]
            candidate.sort()
            if _find_collision_fast(candidate) is None:
                current = candidate
                in_set = set(current)
                not_in_set = [x for x in universe if x not in in_set]
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
                    scaled_c = [_LCM // x for x in current]
                    best_min_diff, _, _ = _min_pairwise_diff(scaled_c)
        
        elif move_type < 0.9:
            # Remove an element (to escape local optima)
            i = rng.randrange(len(current))
            candidate = current[:]
            candidate.pop(i)
            if candidate:
                current = candidate
                in_set = set(current)
                not_in_set = [x for x in universe if x not in in_set]
            else:
                current = []
                in_set = set()
                not_in_set = list(universe)
        
        else:
            # Double swap
            if len(current) >= 2 and len(not_in_set) >= 2:
                i1, i2 = rng.sample(range(len(current)), 2)
                b1, b2 = rng.sample(not_in_set, 2)
                candidate = current[:]
                # Remove higher index first
                idxs = sorted([i1, i2], reverse=True)
                for idx in idxs:
                    candidate.pop(idx)
                candidate.extend([b1, b2])
                candidate.sort()
                if _find_collision_fast(candidate) is None:
                    current = candidate
                    in_set = set(current)
                    not_in_set = [x for x in universe if x not in in_set]
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(current)
                        scaled_c = [_LCM // x for x in current]
                        best_min_diff, _, _ = _min_pairwise_diff(scaled_c)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    universe = list(range(1, 23))
    best = []
    best_len = 0
    
    # Phase 1: Greedy removal with multiple heuristics
    heuristics = ['largest', 'smallest', 'random', 'random', 'largest', 'smallest', 'random']
    for h in heuristics:
        if time.time() > deadline - 30:
            break
        result = _greedy_remove(universe, heuristic=h, deadline=deadline - 30)
        removed = [x for x in universe if x not in result]
        result = _try_add_elements(result, removed, deadline - 25)
        not_in_set = [x for x in universe if x not in result]
        result = _try_swaps(result, not_in_set, deadline - 20)
        if len(result) > best_len:
            best_len = len(result)
            best = result
    
    # Phase 2: Simulated annealing starting from best greedy result
    if time.time() < deadline - 10:
        sa_result = _simulated_annealing(best, deadline - 5)
        if len(sa_result) > best_len:
            best = sa_result
            best_len = len(sa_result)
    
    # Phase 3: Final polish - try adding back any missing elements
    in_set = set(best)
    not_in_set = [x for x in universe if x not in in_set]
    best = _try_add_elements(best, not_in_set, deadline - 2)
    
    # Try one more round of swaps
    not_in_set = [x for x in universe if x not in set(best)]
    if time.time() < deadline - 1:
        best = _try_swaps(best, not_in_set, deadline - 1)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
