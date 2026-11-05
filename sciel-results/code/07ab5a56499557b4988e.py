"""Combined program: fast integer meet-in-the-middle checking + structured
removal / re-addition / 1-swap local-search refinement."""
import time
import math
import random

# Precompute LCM of 1..24
def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for i in range(1, 25):
    _LCM = _lcm(_LCM, i)

# v[i] = LCM / i  (integer representation of 1/i scaled by LCM)
_V = [0] * 25
for i in range(1, 25):
    _V[i] = _LCM // i

_HALF_SIZE = 12
_LEFT = list(range(1, _HALF_SIZE + 1))    # 1..12
_RIGHT = list(range(_HALF_SIZE + 1, 25))  # 13..24

# Precompute all subset sums of reciprocals (scaled) for every 12-bit mask.
_LeftSumsByMask = [None] * 4096
_LeftSumsByMask[0] = {0}
for mask in range(1, 4096):
    # take lowest set bit
    lsb = mask & (-mask)
    i = lsb.bit_length() - 1
    prev = mask ^ lsb
    base = _LeftSumsByMask[prev]
    add = _V[_LEFT[i]]
    s = set(base)
    for x in base:
        s.add(x + add)
    _LeftSumsByMask[mask] = s

_RightSumsByMask = [None] * 4096
_RightSumsByMask[0] = {0}
for mask in range(1, 4096):
    lsb = mask & (-mask)
    i = lsb.bit_length() - 1
    prev = mask ^ lsb
    base = _RightSumsByMask[prev]
    add = _V[_RIGHT[i]]
    s = set(base)
    for x in base:
        s.add(x + add)
    _RightSumsByMask[mask] = s


def _can_form(target, avail_mask):
    """Return True iff some NON-EMPTY subset of elements in avail_mask
    (bit i => element i+1) has reciprocal-sum (scaled) equal to target.
    avail_mask is a 24-bit mask."""
    if target <= 0:
        return False
    left_avail = avail_mask & 0xFFF
    right_avail = (avail_mask >> 12) & 0xFFF
    L_sums = _LeftSumsByMask[left_avail]
    R_sums = _RightSumsByMask[right_avail]
    # Need sL + sR == target with (sL,sR) != (0,0)
    if len(L_sums) < len(R_sums):
        for sL in L_sums:
            needR = target - sL
            if needR in R_sums:
                if sL != 0 or needR != 0:
                    return True
    else:
        for sR in R_sums:
            needL = target - sR
            if needL in L_sums:
                if sR != 0 or needL != 0:
                    return True
    return False


def _check_safe(a_idx, S_mask):
    """True if element a_idx (0-based, element a_idx+1) is safe in S_mask:
    no nonempty subset of S_mask \\ {a_idx} sums (reciprocals) to 1/(a_idx+1)."""
    val = a_idx + 1
    target = _V[val]
    avail_mask = S_mask & ~(1 << a_idx)
    if avail_mask == 0:
        return True
    return not _can_form(target, avail_mask)


def _is_valid(S_mask):
    """True iff every element in S_mask is safe."""
    m = S_mask
    while m:
        lsb = m & (-m)
        i = lsb.bit_length() - 1
        if not _check_safe(i, S_mask):
            return False
        m ^= lsb
    return True


def _greedy_seed(order):
    """Build a valid set by adding elements in `order` (list of 0-based idx)."""
    S = 0
    for idx in order:
        if S & (1 << idx):
            continue
        trial = S | (1 << idx)
        # Correctness: adding element a (value a+1) can only create a violation
        # for element b (value b+1) if 1/b - 1/a > 0 i.e. b < a i.e. b_idx < a_idx.
        ok = _check_safe(idx, trial)
        if ok:
            for i in range(idx):
                if trial >> i & 1:
                    if not _check_safe(i, trial):
                        ok = False
                        break
        if ok:
            S = trial
    return S


def _remove_violators(S_mask, deadline):
    """Iteratively remove violating elements until S is valid."""
    S = S_mask
    changed = True
    while changed and time.time() < deadline:
        changed = False
        m = S
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            if not _check_safe(i, S):
                S ^= lsb
                changed = True
                break  # restart scan after removal
            m ^= lsb
    return S


def _try_add(S_mask, a_idx, deadline):
    """Try to add a_idx to S_mask; return new mask if valid else S_mask."""
    if S_mask & (1 << a_idx):
        return S_mask
    trial = S_mask | (1 << a_idx)
    # check new element
    if not _check_safe(a_idx, trial):
        return S_mask
    # check existing elements with index < a_idx (only those can be newly violated)
    for i in range(a_idx):
        if trial >> i & 1:
            if not _check_safe(i, trial):
                return S_mask
    return trial


def _local_search_1swap(S_mask, deadline):
    """Try 1-swap: remove one, add one (net size same) to reach a better local optimum.
    We maximize cardinality, so a swap only helps if it enables a later add.
    Instead, do: try to add any excluded element; if none, try removing a
    redundant element then re-adding two to net gain. Simpler: do add-then-remove
    repair. Here we just do repeated add attempts in random order."""
    S = S_mask
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # try adding each excluded element (in random order)
        excluded = [i for i in range(24) if not (S >> i & 1)]
        random.shuffle(excluded)
        for idx in excluded:
            if time.time() > deadline:
                break
            newS = _try_add(S, idx, deadline)
            if newS != S:
                S = newS
                improved = True
                break
    return S


def _local_search_2swap(S_mask, deadline):
    """2-swap local search: try removing two elements and adding two others
    to achieve net gain of 0 but potentially enable further additions.
    More aggressive: try all pairs of removals, then try to add back
    excluded elements."""
    S = S_mask
    best_S = S
    best_size = S.bit_count()
    
    if time.time() > deadline:
        return best_S
    
    # Get list of elements in S
    in_S = [i for i in range(24) if (S >> i) & 1]
    not_in_S = [i for i in range(24) if not (S >> i) & 1]
    
    if len(in_S) < 2 or len(not_in_S) < 2:
        return best_S
    
    # Try all pairs to remove
    for i in range(len(in_S)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(in_S)):
            if time.time() > deadline:
                break
            
            a, b = in_S[i], in_S[j]
            # Remove both
            S_trial = S & ~(1 << a) & ~(1 << b)
            
            # Try to add back each excluded element
            for c in not_in_S:
                if time.time() > deadline:
                    break
                newS1 = _try_add(S_trial, c, deadline)
                if newS1 != S_trial:
                    # Now try to add another
                    for d in not_in_S:
                        if d == c:
                            continue
                        if time.time() > deadline:
                            break
                        newS2 = _try_add(newS1, d, deadline)
                        if newS2 != newS1:
                            if newS2.bit_count() > best_size:
                                best_size = newS2.bit_count()
                                best_S = newS2
                            break
                    else:
                        if newS1.bit_count() > best_size:
                            best_size = newS1.bit_count()
                            best_S = newS1
                        continue
                    break
            # Also check if just removing one and adding two helps
            # (handled by single removal in other searches)
    
    return best_S


def _hill_climbing(S_mask, deadline, max_iterations=10000):
    """Aggressive hill climbing with random restarts within the neighborhood.
    Try adding elements, and if stuck, try 2-swaps to escape local optima."""
    S = S_mask
    best_S = S
    best_size = S.bit_count()
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        # First, try to add any excluded element
        excluded = [i for i in range(24) if not (S >> i & 1)]
        random.shuffle(excluded)
        added = False
        for idx in excluded:
            if time.time() > deadline:
                break
            newS = _try_add(S, idx, deadline)
            if newS != S:
                S = newS
                added = True
                if S.bit_count() > best_size:
                    best_size = S.bit_count()
                    best_S = S
                break
        
        if not added:
            # Stuck: try 2-swap to escape
            S_new = _local_search_2swap(S, deadline)
            if S_new.bit_count() > S.bit_count():
                S = S_new
                if S.bit_count() > best_size:
                    best_size = S.bit_count()
                    best_S = S
            else:
                # Try a random perturbation: remove a random element and re-optimize
                if S.bit_count() > 1:
                    in_S = [i for i in range(24) if (S >> i) & 1]
                    random.shuffle(in_S)
                    for rem_idx in in_S[:3]:  # try removing up to 3 random elements
                        if time.time() > deadline:
                            break
                        S_perturbed = S & ~(1 << rem_idx)
                        S_perturbed = _local_search_1swap(S_perturbed, deadline)
                        if S_perturbed.bit_count() > best_size:
                            best_size = S_perturbed.bit_count()
                            best_S = S_perturbed
                        if S_perturbed.bit_count() > S.bit_count():
                            S = S_perturbed
                            break
    
    return best_S


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    best_set = 0
    best_size = 0

    # Phase 0: a deterministic strong seed (decreasing order)
    dec_order = list(range(23, -1, -1))
    S0 = _greedy_seed(dec_order)
    S0 = _remove_violators(S0, deadline)
    S0 = _local_search_1swap(S0, deadline)
    S0 = _hill_climbing(S0, deadline, max_iterations=5000)
    if S0.bit_count() > best_size:
        best_size = S0.bit_count()
        best_set = S0

    # Phase 1: random restarts with greedy seed + refinement
    for attempt in range(50000):
        if time.time() > deadline:
            break

        # random order, occasionally biased
        elems = list(range(24))
        random.shuffle(elems)
        S = _greedy_seed(elems)
        S = _remove_violators(S, deadline)
        S = _local_search_1swap(S, deadline)
        
        # Only do hill climbing if we have time and potential to improve
        if S.bit_count() >= best_size - 1 and time.time() < deadline - 5.0:
            S = _hill_climbing(S, deadline, max_iterations=2000)

        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S

    # Phase 2: from best, try targeted additions of each excluded element
    if time.time() < deadline:
        S = best_set
        S = _hill_climbing(S, deadline, max_iterations=10000)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S

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
