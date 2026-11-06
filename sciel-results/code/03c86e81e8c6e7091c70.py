"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 256
# 1-indexed masks: bit (x-1) corresponds to element x in [1,256]
# mask4[x] = bitmask of elements a (a<x) such that a, a+d, a+2d, a+3d=x is a 4-AP
#           i.e., a = x-3d for d=1..(x-1)//3  ->  a in {x-3, x-6, x-9, ...}
_MASK4 = [0] * (_N + 1)
for x in range(1, _N + 1):
    m = 0
    d = 1
    while x - 3 * d >= 1:
        a = x - 3 * d
        m |= 1 << (a - 1)
        d += 1
    _MASK4[x] = m

# mask3[x] = bitmask of elements a (a<x) such that a, m, x form a 3-AP
#           i.e., a = 2m - x for m=(x+2)//2..x-1  -> a = x-2, x-4, ...
_MASK3 = [0] * (_N + 1)
for x in range(1, _N + 1):
    m = 0
    d = 1
    while x - 2 * d >= 1:
        a = x - 2 * d
        m |= 1 << (a - 1)
        d += 1
    _MASK3[x] = m


def _count_3aps(mask):
    """Count distinct 3-APs (positive diff) in the set represented by mask."""
    cnt = 0
    mm = mask
    while mm:
        lsb = mm & (-mm)
        x = lsb.bit_length()
        cnt += (mask & _MASK3[x]).bit_count()
        mm ^= lsb
    return cnt


def _can_add(mask, x):
    """True if adding x creates no nonconstant 4-AP."""
    return (mask & _MASK4[x]) == 0


def _greedy(seed=None):
    """Greedy first-fit in ascending order."""
    mask = 0
    for x in range(1, _N + 1):
        if _can_add(mask, x):
            mask |= 1 << (x - 1)
    return mask


def _local_search(mask, time_budget, rng):
    """Improve mask via random swap moves that keep no-4-AP and increase 3-AP count."""
    base_score = _count_3aps(mask)
    deadline = time.time() + time_budget
    best_mask = mask
    best_score = base_score
    cur_mask = mask
    cur_score = base_score

    # Precompute element list and a membership helper via mask bits
    # Swap: remove a (in set), add b (not in set) if no 4-AP and score improves
    while time.time() < deadline:
        # pick a random element to remove
        in_bits = cur_mask
        if in_bits == 0:
            break
        a_bit = (in_bits >> rng.randrange(in_bits.bit_length())) & 1
        # find a set bit
        # pick random set bit
        pos = rng.randrange(cur_mask.bit_length())
        a_bit = (cur_mask >> pos) & 1
        if a_bit == 0:
            # find a set bit
            tmp = cur_mask
            # random set bit
            k = rng.randrange(tmp.bit_count())
            t = tmp
            while k > 0:
                c = t & (-t)
                t ^= c
                k -= 1
            a_bit = c
        a = a_bit.bit_length()  # element value
        # candidate b to add
        # try several random b
        improved = False
        for _ in range(20):
            b = rng.randrange(1, _N + 1)
            b_bit = 1 << (b - 1)
            if cur_mask & b_bit:
                continue
            # after removing a, can we add b?
            m2 = cur_mask & ~a_bit
            if (m2 & _MASK4[b]) != 0:
                continue
            new_mask = m2 | b_bit
            # score delta
            # removing a loses 3-APs containing a; adding b gains 3-APs with b
            # compute quickly:
            delta = _count_3aps(new_mask) - cur_score
            if delta > 0:
                cur_mask = new_mask
                cur_score += delta
                improved = True
                if cur_score > best_score:
                    best_score = cur_score
                    best_mask = cur_mask
                break
        if not improved:
            # occasional restart to avoid local trap
            if rng.random() < 0.05:
                # keep best, restart from it
                cur_mask = best_mask
                cur_score = best_score
    return best_mask, best_score


def construct():
    t0 = time.time()
    hard_deadline = t0 + 80.0  # safety margin before 90s kill

    rng = random.Random(12345)

    # Start from greedy
    best_mask = _greedy()
    best_score = _count_3aps(best_mask)

    # Local search with the bulk of the budget
    ls_budget = 70.0
    best_mask, best_score = _local_search(best_mask, ls_budget, rng)

    # If time remains, try a few more independent local-search rounds from greedy
    while time.time() < hard_deadline - 5.0:
        m = _greedy()
        m, s = _local_search(m, 5.0, rng)
        if s > best_score:
            best_score = s
            best_mask = m

    # Convert mask to sorted list
    out = []
    mm = best_mask
    while mm:
        lsb = mm & (-mm)
        out.append(lsb.bit_length())
        mm ^= lsb
    out.sort()
    if not out:
        out = [1]
    return out
# EVOLVE-BLOCK-END
