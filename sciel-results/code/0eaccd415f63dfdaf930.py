"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48

_pair_res = None

def _precompute():
    """Precompute pair residue data: for each a, map residue -> list of (b,c,bitmask)."""
    global _pair_res
    if _pair_res is not None:
        return _pair_res
    pr = {}
    for a in range(1, _N + 1):
        pr[a] = {}
        for b in range(1, _N + 1):
            if b == a:
                continue
            for c in range(b + 1, _N + 1):
                if c == a:
                    continue
                r = (b + c) % a
                bm = (1 << (b - 1)) | (1 << (c - 1))
                if r not in pr[a]:
                    pr[a][r] = []
                pr[a][r].append((b, c, bm))
    _pair_res = pr
    return _pair_res

def _get_pair_res():
    _precompute()
    return _pair_res

def _can_add(x, selected_mask, selected_list):
    """Check if x can be added to selected without violation."""
    pr = _get_pair_res()
    n = len(selected_list)
    
    # Check 1: x divides sum of any 3 distinct elements in selected
    if n >= 3:
        for i in range(n):
            si = selected_list[i]
            for j in range(i + 1, n):
                sj = selected_list[j]
                s_ij = si + sj
                for k in range(j + 1, n):
                    if (s_ij + selected_list[k]) % x == 0:
                        return False
    
    # Check 2: For each a in selected, a | (x + b + c) for some b,c in selected\{a}
    x_bit = 1 << (x - 1)
    for a in selected_list:
        target_res = (-x) % a
        if target_res in pr[a]:
            for (b, c, bm) in pr[a][target_res]:
                if (bm & selected_mask) == bm:
                    return False
    
    return True

def _is_valid(selected):
    """Check if the set is valid."""
    n = len(selected)
    if n < 4:
        return True
    for a in selected:
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _compute_degrees(selected_set, remaining_set):
    """Compute conflict degree for each remaining element.
    Degree = number of hyperedges (a,b,c,d) where a divides b+c+d,
    with a in selected or a==x, and b,c,d in remaining or selected.
    Approximation: for each x in remaining, count how many a in selected
    would have a | (x + b + c) for some b,c in remaining, plus
    how many triplets in remaining sum to multiple of x."""
    pr = _get_pair_res()
    degrees = {}
    sel_list = list(selected_set)
    rem_list = list(remaining_set)
    
    for x in rem_list:
        deg = 0
        # Count conflicts with selected elements
        for a in sel_list:
            target = (-x) % a
            if target in pr[a]:
                for (b, c, bm) in pr[a][target]:
                    if b in remaining_set and c in remaining_set:
                        deg += 1
                        break  # count at most 1 per (a, x)
        # Count triplets in remaining divisible by x
        if len(rem_list) >= 3:
            for i in range(len(rem_list)):
                if rem_list[i] == x:
                    continue
                ri = rem_list[i]
                for j in range(i + 1, len(rem_list)):
                    if rem_list[j] == x:
                        continue
                    rj = rem_list[j]
                    for k in range(j + 1, len(rem_list)):
                        if rem_list[k] == x:
                            continue
                        rk = rem_list[k]
                        if (ri + rj + rk) % x == 0:
                            deg += 1
                            break  # count at most 1 per x
        degrees[x] = deg
    
    return degrees

def _greedy_degree_based(seed=0):
    """Greedy that picks element with lowest conflict degree first."""
    _get_pair_res()
    rng = random.Random(seed)
    selected = []
    selected_set = set()
    selected_mask = 0
    remaining = set(range(1, _N + 1))
    
    # Initial ordering: start with larger elements (less likely to divide sums)
    # But use degree to break ties and guide selection
    
    while remaining and len(selected) < _MAX:
        if not remaining:
            break
        
        # Compute degrees for remaining elements
        degrees = _compute_degrees(selected_set, remaining)
        
        # Pick element with lowest degree, tie-break by larger value
        best_x = None
        best_key = None
        for x in remaining:
            key = (degrees[x], -x)  # minimize degree, then maximize x
            if best_key is None or key < best_key:
                best_key = key
                best_x = x
        
        if best_x is None:
            break
        
        if _can_add(best_x, selected_mask, selected):
            selected.append(best_x)
            selected_set.add(best_x)
            selected_mask |= (1 << (best_x - 1))
            remaining.remove(best_x)
        else:
            # Try next best
            found = False
            for x in sorted(remaining, key=lambda v: (degrees[v], -v)):
                if x == best_x:
                    continue
                if _can_add(x, selected_mask, selected):
                    selected.append(x)
                    selected_set.add(x)
                    selected_mask |= (1 << (x - 1))
                    remaining.remove(x)
                    found = True
                    break
            if not found:
                break
    
    return selected

def _greedy_largest_first():
    """Simple largest-first greedy."""
    _get_pair_res()
    selected = []
    selected_mask = 0
    for x in range(_N, 0, -1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected_mask, selected):
            selected.append(x)
            selected_mask |= (1 << (x - 1))
    return selected

def _greedy_smallest_first():
    """Simple smallest-first greedy."""
    _get_pair_res()
    selected = []
    selected_mask = 0
    for x in range(1, _N + 1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected_mask, selected):
            selected.append(x)
            selected_mask |= (1 << (x - 1))
    return selected

def _greedy_random(seed):
    """Greedy with random ordering."""
    _get_pair_res()
    rng = random.Random(seed)
    selected = []
    selected_mask = 0
    candidates = list(range(1, _N + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected_mask, selected):
            selected.append(x)
            selected_mask |= (1 << (x - 1))
    return selected

def _local_search(selected, time_limit=10.0):
    """Enhanced local search with add, 1-for-1 swap, and 2-for-2 swap."""
    if not _is_valid(selected):
        return selected
    
    start = time.time()
    best = list(selected)
    best_set = set(best)
    best_mask = 0
    for v in best:
        best_mask |= (1 << (v - 1))
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, _N + 1) if x not in best_set]
        if not candidates:
            break
        
        # Try adding
        for x in candidates:
            if _can_add(x, best_mask, best):
                best.append(x)
                best_set.add(x)
                best_mask |= (1 << (x - 1))
                improved = True
                break
        
        if not improved and len(best) >= 1:
            # Try 1-for-1 swap
            swap_done = False
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                removed = best[i]
                temp_mask = best_mask & ~(1 << (removed - 1))
                temp_list = best[:i] + best[i + 1:]
                
                for x in candidates:
                    if x in best_set:
                        continue
                    if _can_add(x, temp_mask, temp_list):
                        best = temp_list + [x]
                        best_set = best_set - {removed} | {x}
                        best_mask = temp_mask | (1 << (x - 1))
                        swap_done = True
                        break
                
                if swap_done:
                    break
        
        if not improved and not swap_done:
            # Try 2-for-2 swap
            found = False
            if len(best) >= 2:
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        break
                    for j in range(i + 1, len(best)):
                        r1, r2 = best[i], best[j]
                        temp_mask = best_mask & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))
                        temp_list = [v for k, v in enumerate(best) if k != i and k != j]
                        
                        added_count = 0
                        for x in candidates:
                            if x in best_set:
                                continue
                            if _can_add(x, temp_mask, temp_list):
                                temp_list.append(x)
                                temp_mask |= (1 << (x - 1))
                                added_count += 1
                                if added_count == 2:
                                    break
                        
                        if added_count > 0:
                            best = temp_list
                            best_set = set(best)
                            best_mask = temp_mask
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                break
    
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin
    
    # Precompute
    _get_pair_res()
    
    result = []
    
    # Largest-first greedy
    selected = _greedy_largest_first()
    result = selected
    
    # Smallest-first greedy
    selected = _greedy_smallest_first()
    if len(selected) > len(result):
        result = selected
    
    # Degree-based greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 25:
            break
        selected = _greedy_degree_based(seed)
        if len(selected) > len(result):
            result = selected
    
    # Random greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        selected = _greedy_random(seed)
        if len(selected) > len(result):
            result = selected
    
    # Local search on the best result
    if time.time() < deadline - 5:
        remaining_time = min(12.0, deadline - time.time() - 2.0)
        if remaining_time > 1.0:
            result = _local_search(result, time_limit=remaining_time)
    
    # Final validation
    if not _is_valid(result):
        # Fallback: rebuild greedily
        result = _greedy_largest_first()
    
    return result
# EVOLVE-BLOCK-END
