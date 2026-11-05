"""Combined program: integer LCM-based meet-in-the-middle with multi-strategy search."""
import time
import math
import random

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for i in range(1, 25):
    _LCM = _lcm(_LCM, i)

# v[i] = LCM / i
_V = [0] * 25
for i in range(1, 25):
    _V[i] = _LCM // i

_HALF_SIZE = 12
_LEFT = list(range(1, _HALF_SIZE + 1))   # 1..12
_RIGHT = list(range(_HALF_SIZE + 1, 25)) # 13..24

# Precompute LeftSumsByMask and RightSumsByMask
_LeftSumsByMask = [set() for _ in range(4096)]
for mask in range(4096):
    sub = mask
    while sub:
        s = 0
        for i in range(12):
            if sub >> i & 1:
                s += _V[_LEFT[i]]
        _LeftSumsByMask[mask].add(s)
        sub = (sub - 1) & mask
    _LeftSumsByMask[mask].add(0)

_RightSumsByMask = [set() for _ in range(4096)]
for mask in range(4096):
    sub = mask
    while sub:
        s = 0
        for i in range(12):
            if sub >> i & 1:
                s += _V[_RIGHT[i]]
        _RightSumsByMask[mask].add(s)
        sub = (sub - 1) & mask
    _RightSumsByMask[mask].add(0)

def _check_safe_fast(a_idx, S_mask):
    """Check if element a_idx (0-based) in set S_mask is safe."""
    val = a_idx + 1
    target = _V[val]
    avail_mask = S_mask & ~(1 << a_idx)
    
    if avail_mask == 0:
        return True
    
    left_avail = avail_mask & 0xFFF
    right_avail = (avail_mask >> 12) & 0xFFF
    
    L_sums = _LeftSumsByMask[left_avail]
    R_sums = _RightSumsByMask[right_avail]
    
    # Check if target can be formed by sL + sR
    if len(L_sums) < len(R_sums):
        for sL in L_sums:
            needR = target - sL
            if needR in R_sums:
                if sL != 0 or needR != 0:
                    return False
    else:
        for sR in R_sums:
            needL = target - sR
            if needL in L_sums:
                if sR != 0 or needL != 0:
                    return False
                    
    return True

def _is_valid_set(S_mask):
    """Check if the entire set is valid (no reciprocal subsum violations)."""
    for i in range(24):
        if S_mask >> i & 1:
            if not _check_safe_fast(i, S_mask):
                return False
    return True

def _greedy_build(order, deadline):
    """Greedy construction following a given element order (list of 0-based indices)."""
    S = 0
    for idx in order:
        if time.time() > deadline:
            break
        trial = S | (1 << idx)
        ok = True
        # Only need to check the new element and potentially affected existing elements
        # Actually, adding an element can only create NEW violations involving that element
        # or violations where the new element is part of a subset sum for another element
        # So we need to check: is the new element safe? AND are all existing elements still safe?
        # The new element being in the "others" for existing elements could create violations.
        # So we must check all elements in trial.
        for i in range(24):
            if trial >> i & 1:
                if not _check_safe_fast(i, trial):
                    ok = False
                    break
        if ok:
            S = trial
    return S

def _local_search(S_mask, deadline, max_iters=200):
    """Try to improve the set by swapping or adding/removing elements."""
    best = S_mask
    best_size = S_mask.bit_count()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding each element not in the set
        for i in range(24):
            if time.time() > deadline:
                break
            if not (best >> i & 1):
                trial = best | (1 << i)
                if _is_valid_set(trial):
                    best = trial
                    best_size += 1
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try removing one element and adding two others
        for i in range(24):
            if time.time() > deadline:
                break
            if best >> i & 1:
                base = best & ~(1 << i)
                # Try adding two elements not in base
                for j in range(24):
                    if time.time() > deadline:
                        break
                    if not (base >> j & 1):
                        for k in range(j + 1, 24):
                            if time.time() > deadline:
                                break
                            if not (base >> k & 1):
                                trial = base | (1 << j) | (1 << k)
                                if trial.bit_count() > best_size and _is_valid_set(trial):
                                    best = trial
                                    best_size = trial.bit_count()
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def construct():
    """Return a nonempty subset of [1,24] maximizing cardinality."""
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill
    
    best_set = 0
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        sz = S.bit_count()
        if sz > best_size:
            best_set = S
            best_size = sz
    
    # Strategy 1: Descending order (large elements first)
    order_desc = list(range(23, -1, -1))
    S = _greedy_build(order_desc, deadline)
    update_best(S)
    
    if time.time() > deadline:
        pass  # fall through to return
    
    # Strategy 2: Ascending order
    order_asc = list(range(24))
    S = _greedy_build(order_asc, deadline)
    update_best(S)
    
    # Strategy 3: Alternating large/small
    order_alt = []
    large = list(range(23, 11, -1))  # 24 down to 13
    small = list(range(11, -1, -1))   # 12 down to 1
    for i in range(12):
        if i < len(large):
            order_alt.append(large[i])
        if i < len(small):
            order_alt.append(small[i])
    S = _greedy_build(order_alt, deadline)
    update_best(S)
    
    # Strategy 4: Random restarts with local search
    random.seed(42)
    for attempt in range(3000):
        if time.time() > deadline:
            break
        
        elems = list(range(24))
        random.shuffle(elems)
        
        S = _greedy_build(elems, deadline)
        if S.bit_count() >= best_size:
            # Try local search to improve
            S2 = _local_search(S, deadline, max_iters=50)
            update_best(S2)
    
    # Final: try all single deterministic orderings if time permits
    if time.time() < deadline - 5:
        # Try a few more structured orderings
        orderings = [
            list(range(23, -1, -1)),
            list(range(24)),
            [23, 12, 22, 11, 21, 10, 20, 9, 19, 8, 18, 7, 17, 6, 16, 5, 15, 4, 14, 3, 13, 2, 0, 1],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ]
        for order in orderings:
            if time.time() > deadline:
                break
            S = _greedy_build(order, deadline)
            update_best(S)
            if S.bit_count() > best_size:
                S2 = _local_search(S, deadline, max_iters=100)
                update_best(S2)
    
    if best_set != 0:
        result = [i + 1 for i in range(24) if best_set >> i & 1]
    else:
        result = [1]
        
    return result

if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Result: {r}, size={len(r)}, time={t1-t0:.2f}s")
