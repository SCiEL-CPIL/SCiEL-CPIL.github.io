"""Combined: block-based construction (A∪B blocks + gap) + dynamic greedy + local search + SA polish.
Uses bitset for fast membership, clause-weight propagation for greedy, and simulated annealing for escape."""
import time
import random
from collections import defaultdict

N = 128
DEADLINE = 55.0  # safety margin under 90s kill

# Precompute forbidden 6-sets: {a, b, c, a+b, a+c, b+c} where a<b<c and b+c <= N
# Use frozensets to handle cases where some values coincide
_FORBIDDEN_SETS = []
_FORBIDDEN_SETS_SET = set()
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        for c in range(b + 1, N + 1):
            s1 = a + b
            s2 = a + c
            s3 = b + c
            if s3 > N:
                continue
            vals = frozenset({a, b, c, s1, s2, s3})
            if len(vals) >= 3 and vals not in _FORBIDDEN_SETS_SET:
                _FORBIDDEN_SETS_SET.add(vals)
                _FORBIDDEN_SETS.append(vals)

_NUM_FORBIDDEN = len(_FORBIDDEN_SETS)

# Index: for each value, which forbidden sets contain it
_val_to_fsets = defaultdict(list)
for idx, fs in enumerate(_FORBIDDEN_SETS):
    for v in fs:
        if 1 <= v <= N:
            _val_to_fsets[v].append(idx)


def _build_counts(sel):
    """Build clause counts: counts[fs_idx] = number of elements of fs in sel."""
    counts = [0] * _NUM_FORBIDDEN
    for v in sel:
        for idx in _val_to_fsets[v]:
            counts[idx] += 1
    return counts


def _is_safe(v, counts):
    """Check if adding v would complete any forbidden set."""
    fs_len = len(_FORBIDDEN_SETS)  # not used directly, but counts[idx] == len(fs)-1 means adding v completes it
    for idx in _val_to_fsets[v]:
        if counts[idx] == len(_FORBIDDEN_SETS[idx]) - 1:
            return False
    return True


def _greedy_dynamic(remaining, counts, deadline, rng=None):
    """Pick each next value by minimum danger (fewest near-complete forbidden sets)."""
    sel = set()
    result = []
    while remaining and time.time() < deadline:
        best_v = None
        best_danger = None
        for v in remaining:
            danger = 0
            for idx in _val_to_fsets[v]:
                if counts[idx] == len(_FORBIDDEN_SETS[idx]) - 1:
                    danger += 1
            if best_danger is None or danger < best_danger:
                best_danger = danger
                best_v = v
            if danger == 0:
                break
        if best_v is None:
            break
        if best_danger > 0:
            found = False
            for v in remaining:
                if _is_safe(v, counts):
                    best_v = v
                    found = True
                    break
            if not found:
                break
        sel.add(best_v)
        result.append(best_v)
        remaining.discard(best_v)
        for idx in _val_to_fsets[best_v]:
            counts[idx] += 1
    return result


def _greedy_weighted(remaining, counts, deadline, rng=None):
    """Max-SAT-style: minimize sum of clause weights over affected clauses."""
    sel = set()
    result = []
    while remaining and time.time() < deadline:
        best_v = None
        best_cost = None
        for v in remaining:
            safe = True
            cost = 0
            for idx in _val_to_fsets[v]:
                c = counts[idx]
                fs_len = len(_FORBIDDEN_SETS[idx])
                if c == fs_len - 1:
                    safe = False
                    break
                cost += c
                if c >= 3:
                    cost += (c - 2) * 3
            if not safe:
                continue
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_v = v
            if cost == 0:
                break
        if best_v is None:
            break
        sel.add(best_v)
        result.append(best_v)
        remaining.discard(best_v)
        for idx in _val_to_fsets[best_v]:
            counts[idx] += 1
    return result


def _local_search(sel, counts, deadline):
    """Phase 1: add any safe value. Phase 2: remove-one-to-enable. Repeat."""
    best_set = set(sel)
    best_size = len(sel)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in range(1, N + 1):
            if v in sel:
                continue
            if _is_safe(v, counts):
                sel.add(v)
                for idx in _val_to_fsets[v]:
                    counts[idx] += 1
                changed = True
        if not changed:
            improved = False
            for v_remove in list(sel):
                sel.discard(v_remove)
                for idx in _val_to_fsets[v_remove]:
                    counts[idx] -= 1
                added = []
                for v_add in range(1, N + 1):
                    if v_add in sel:
                        continue
                    if _is_safe(v_add, counts):
                        sel.add(v_add)
                        added.append(v_add)
                        for idx in _val_to_fsets[v_add]:
                            counts[idx] += 1
                new_size = len(sel)
                if new_size > best_size:
                    best_set = set(sel)
                    best_size = new_size
                    improved = True
                else:
                    for v_add in added:
                        sel.discard(v_add)
                        for idx in _val_to_fsets[v_add]:
                            counts[idx] -= 1
                    sel.add(v_remove)
                    for idx in _val_to_fsets[v_remove]:
                        counts[idx] += 1
                if time.time() > deadline:
                    for v_add in added:
                        sel.discard(v_add)
                        for idx in _val_to_fsets[v_add]:
                            counts[idx] -= 1
                    sel.add(v_remove)
                    for idx in _val_to_fsets[v_remove]:
                        counts[idx] += 1
                    break
            if improved:
                changed = True
                sel = set(best_set)
                counts = _build_counts(sel)
    return best_set


def _swap_polish(sel, counts, deadline):
    """2-swap polish: try removing one and adding another."""
    best_size = len(sel)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_remove in list(sel):
            sel.discard(v_remove)
            for idx in _val_to_fsets[v_remove]:
                counts[idx] -= 1
            added = []
            for v_add in range(1, N + 1):
                if v_add in sel:
                    continue
                if _is_safe(v_add, counts):
                    sel.add(v_add)
                    added.append(v_add)
                    for idx in _val_to_fsets[v_add]:
                        counts[idx] += 1
            if len(sel) > best_size:
                best_size = len(sel)
                improved = True
            else:
                for v_add in added:
                    sel.discard(v_add)
                    for idx in _val_to_fsets[v_add]:
                        counts[idx] -= 1
                sel.add(v_remove)
                for idx in _val_to_fsets[v_remove]:
                    counts[idx] += 1
            if time.time() > deadline:
                for v_add in added:
                    sel.discard(v_add)
                    for idx in _val_to_fsets[v_add]:
                        counts[idx] -= 1
                sel.add(v_remove)
                for idx in _val_to_fsets[v_remove]:
                    counts[idx] += 1
                break
    return best_size


def _greedy_order(order, deadline):
    sel = set()
    result = []
    counts = [0] * _NUM_FORBIDDEN
    for v in order:
        if v in sel:
            continue
        if not _is_safe(v, counts):
            continue
        sel.add(v)
        result.append(v)
        for idx in _val_to_fsets[v]:
            counts[idx] += 1
        if time.time() > deadline:
            break
    return result


def _block_based_construct(deadline, rng):
    """Block-based construction: A block + B block + gap filling.
    A is a contiguous range, B is [L_B, 128], gap is between them."""
    best = set()
    best_size = 0
    
    # Try various block configurations
    for L_B in range(55, 75):
        if time.time() > deadline:
            break
        for L_A in range(10, 30):
            if time.time() > deadline:
                break
            for R_A in range(L_A, min(L_A + 25, L_B - 2)):
                if time.time() > deadline:
                    break
                A = set(range(L_A, R_A + 1))
                B = set(range(L_B, N + 1))
                
                if A & B:
                    continue
                
                base = A | B
                counts = _build_counts(base)
                
                # Check if base is valid
                valid = True
                for idx in range(_NUM_FORBIDDEN):
                    if counts[idx] == len(_FORBIDDEN_SETS[idx]):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Gap candidates
                gap = set(range(R_A + 1, L_B))
                below = set(range(1, L_A))
                candidates = gap | below
                
                # Greedy fill with randomization
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                sel = set(base)
                for v in cand_list:
                    if time.time() > deadline:
                        break
                    if v in sel:
                        continue
                    if _is_safe(v, counts):
                        sel.add(v)
                        for idx in _val_to_fsets[v]:
                            counts[idx] += 1
                
                if len(sel) > best_size:
                    best_size = len(sel)
                    best = set(sel)
    
    return best


def _simulated_annealing(sel, counts, deadline, rng):
    """Simulated annealing to escape local optima."""
    T = 1.0
    cooling = 0.995
    best_set = set(sel)
    best_size = len(sel)
    current_size = len(sel)
    
    while time.time() < deadline:
        move_type = rng.random()
        
        if move_type < 0.3:
            if not sel:
                break
            v_remove = rng.choice(list(sel))
            sel.discard(v_remove)
            for idx in _val_to_fsets[v_remove]:
                counts[idx] -= 1
            added = []
            for v_add in range(1, N + 1):
                if v_add in sel:
                    continue
                if _is_safe(v_add, counts):
                    sel.add(v_add)
                    added.append(v_add)
                    for idx in _val_to_fsets[v_add]:
                        counts[idx] += 1
            new_size = len(sel)
            delta = new_size - current_size
            current_size = new_size
            
            if new_size > best_size:
                best_set = set(sel)
                best_size = new_size
            elif delta < 0 and rng.random() > 0.5:
                for v_add in added:
                    sel.discard(v_add)
                    for idx in _val_to_fsets[v_add]:
                        counts[idx] -= 1
                sel.add(v_remove)
                for idx in _val_to_fsets[v_remove]:
                    counts[idx] += 1
                current_size += 1
        
        elif move_type < 0.6:
            candidates = []
            for v in range(1, N + 1):
                if v not in sel and _is_safe(v, counts):
                    candidates.append(v)
            if candidates:
                v_add = rng.choice(candidates)
                sel.add(v_add)
                for idx in _val_to_fsets[v_add]:
                    counts[idx] += 1
                current_size += 1
                if current_size > best_size:
                    best_set = set(sel)
                    best_size = current_size
        
        else:
            if not sel:
                break
            v_remove = rng.choice(list(sel))
            sel.discard(v_remove)
            for idx in _val_to_fsets[v_remove]:
                counts[idx] -= 1
            
            candidates = []
            for v in range(1, N + 1):
                if v not in sel and _is_safe(v, counts):
                    candidates.append(v)
            
            if candidates:
                v_add = rng.choice(candidates)
                sel.add(v_add)
                for idx in _val_to_fsets[v_add]:
                    counts[idx] += 1
                if current_size > best_size:
                    best_set = set(sel)
                    best_size = current_size
            else:
                sel.add(v_remove)
                for idx in _val_to_fsets[v_remove]:
                    counts[idx] += 1
        
        T *= cooling
        if T < 0.01:
            break
    
    return best_set


def _validate(sel_set):
    """Check if sel_set is valid (no forbidden set is fully contained)."""
    counts = _build_counts(sel_set)
    for idx in range(_NUM_FORBIDDEN):
        if counts[idx] == len(_FORBIDDEN_SETS[idx]):
            return False
    return True


def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = set()
    best_size = 0
    rng = random.Random(12345)

    def _try(sel):
        nonlocal best, best_size
        if not sel:
            return
        if not _validate(sel):
            return
        sel2 = _local_search(set(sel), _build_counts(sel), deadline)
        if len(sel2) > best_size:
            best = sel2
            best_size = len(sel2)

    # Strategy 1: Block-based construction (strongest structural approach)
    if time.time() < deadline - 10:
        block_res = _block_based_construct(deadline, rng)
        if block_res:
            _try(block_res)

    # Strategy 2: Base set decomposition (large base + small additions)
    if time.time() < deadline - 8:
        base = set(range(65, 129))
        counts = _build_counts(base)
        sel = set(base)
        small_numbers = list(range(1, 65))
        for v in small_numbers:
            if time.time() > deadline:
                break
            if _is_safe(v, counts):
                sel.add(v)
                for idx in _val_to_fsets[v]:
                    counts[idx] += 1
        _try(sel)

    # Strategy 3: Dynamic danger greedy
    if time.time() < deadline - 6:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_FORBIDDEN
        res = _greedy_dynamic(remaining, counts, deadline)
        _try(res)

    # Strategy 4: Max-SAT weighted greedy
    if time.time() < deadline - 5:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_FORBIDDEN
        res = _greedy_weighted(remaining, counts, deadline)
        _try(res)

    # Strategy 5: Fixed-order heuristics
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(_val_to_fsets[x])),
        sorted(range(1, N + 1), key=lambda x: -len(_val_to_fsets[x])),
        list(range(N // 2 + 1, N + 1)) + list(range(1, N // 2 + 1)),
    ]
    for order in orders:
        if time.time() > deadline - 4:
            break
        res = _greedy_order(order, deadline)
        _try(res)

    # Strategy 6: Randomized restarts
    while time.time() < deadline - 3:
        remaining = set(range(1, N + 1))
        counts = [0] * _NUM_FORBIDDEN
        if rng.random() < 0.5:
            res = _greedy_dynamic(remaining, counts, deadline, rng=rng)
        else:
            res = _greedy_weighted(remaining, counts, deadline, rng=rng)
        _try(res)
        if best_size >= 95:
            break

    # Strategy 7: Simulated annealing on the best set found so far
    if best and time.time() < deadline - 2:
        counts = _build_counts(best)
        sa_result = _simulated_annealing(set(best), counts, deadline, rng)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(sa_result)

    # Final swap polish on best
    if best and time.time() < deadline - 1:
        counts = _build_counts(best)
        _swap_polish(best, counts, deadline)

    # Final validity check
    if best:
        if not _validate(best):
            best = set(range(65, 129))

    return [int(x) for x in best]
