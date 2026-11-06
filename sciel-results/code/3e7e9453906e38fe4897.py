import time
import random
import math

TARGET = 200
N = 100
_SUM_MASK = (1 << (TARGET + 1)) - 1  # keep only sums 0..200


def _popcount(m):
    return bin(m).count("1")


def _is_200_free(mask):
    """Return True iff no submultiset of the chosen values sums to TARGET.
    Values are 1..N, bit (v-1) set means value v is chosen.
    """
    if mask == 0:
        return False  # must be nonempty
    reach = 1  # bit 0 set (sum 0)
    # iterate only set bits
    mm = mask
    while mm:
        low = mm & (-mm)
        v = low.bit_length()  # value = bit_index + 1
        reach = (reach | (reach << v)) & _SUM_MASK
        if (reach >> TARGET) & 1:
            return False
        mm ^= low
    return True


def _compute_reach(mask):
    """Compute the reachable sum bitset for a given mask."""
    reach = 1
    mm = mask
    while mm:
        low = mm & (-mm)
        v = low.bit_length()
        reach = (reach | (reach << v)) & _SUM_MASK
        mm ^= low
    return reach


def _cardinality(mask):
    return _popcount(mask)


def _flip(mask, i):
    return mask ^ (1 << i)


def _add_value(mask, v):
    return mask | (1 << (v - 1))


def _remove_value(mask, v):
    return mask & ~(1 << (v - 1))


def _set_bits(mask):
    """Return list of values present in mask."""
    vals = []
    mm = mask
    while mm:
        low = mm & (-mm)
        v = low.bit_length()
        vals.append(v)
        mm ^= low
    return vals


def _init_interval(lo, hi):
    m = 0
    for v in range(lo, hi + 1):
        m |= (1 << (v - 1))
    return m


def _init_greedy_desc():
    selected = 0
    reach = 1
    for x in range(100, 0, -1):
        new = (reach | (reach << x)) & _SUM_MASK
        if not ((new >> TARGET) & 1):
            selected |= (1 << (x - 1))
            reach = new
    return selected


def _init_greedy_asc():
    selected = 0
    reach = 1
    for x in range(1, 101):
        new = (reach | (reach << x)) & _SUM_MASK
        if not ((new >> TARGET) & 1):
            selected |= (1 << (x - 1))
            reach = new
    return selected


def _init_random(rng, p=0.5):
    m = 0
    for v in range(1, N + 1):
        if rng.random() < p:
            m |= (1 << (v - 1))
    return m


def _repair(mask):
    """Drop elements until the set is 200-free (greedy, drop largest first).
    Returns a feasible (possibly empty) mask."""
    m = mask
    if m == 0:
        return m
    if _is_200_free(m):
        return m
    while m and not _is_200_free(m):
        hi = m.bit_length() - 1
        m &= ~(1 << hi)
    return m


def _sa(mask, rng, t0, t1, iters, deadline):
    cur = mask
    cur_c = _cardinality(cur)
    best = mask
    best_c = cur_c
    cur = _repair(cur)
    cur_c = _cardinality(cur)
    best = cur
    best_c = cur_c
    
    # Maintain reachable sums
    cur_reach = _compute_reach(cur)
    
    T = t0
    for i in range(iters):
        if time.time() > deadline:
            break
        T = t0 + (t1 - t0) * (i / max(1, iters - 1))
        
        # propose flip
        v = rng.randint(1, N)
        idx = v - 1
        bit = 1 << idx
        
        if cur & bit:
            # Remove v
            cand = cur ^ bit
            # Recompute reach for candidate
            cand_reach = _compute_reach(cand)
            cand_ok = not ((cand_reach >> TARGET) & 1)
            cand_c = cur_c - 1
        else:
            # Add v
            cand = cur | bit
            # Incremental update
            cand_reach = (cur_reach | (cur_reach << v)) & _SUM_MASK
            cand_ok = not ((cand_reach >> TARGET) & 1)
            cand_c = cur_c + 1
        
        if cand_ok:
            d = cand_c - cur_c  # positive = better (more elements)
            if d >= 0 or (T > 0 and rng.random() < math.exp(d / T)):
                cur = cand
                cur_c = cand_c
                cur_reach = cand_reach
                if cur_c > best_c:
                    best = cur
                    best_c = cur_c
    return best, best_c


def construct():
    start = time.time()
    HARD_DEADLINE = start + 85.0  # safety margin under 90s kill

    rng = random.Random(12345)

    # candidate initial solutions
    inits = [
        _init_interval(66, 100),
        _init_greedy_desc(),
        _init_greedy_asc(),
        _init_random(rng, 0.5),
        _init_random(rng, 0.6),
        _init_interval(60, 100),
        _init_interval(63, 100),
        _init_interval(50, 100),
        _init_interval(40, 100),
    ]

    # take best feasible initial
    best_mask = 0
    best_c = 0
    for m in inits:
        m = _repair(m)
        c = _cardinality(m)
        if c > best_c:
            best_c = c
            best_mask = m

    # annealing budget: run SA from the best initial and a few restarts
    # each run uses a fraction of the remaining time
    def time_left():
        return HARD_DEADLINE - time.time()

    # First SA from best initial with the bulk of the budget
    # Increase iterations since SA is faster now
    iters1 = 500000
    m, c = _sa(best_mask, rng, t0=2.0, t1=0.01, iters=iters1,
               deadline=HARD_DEADLINE)
    if c > best_c:
        best_c = c
        best_mask = m

    # If time remains, do extra restarts from random inits (SA)
    guard = 0
    while time_left() > 6.0 and guard < 100:
        guard += 1
        rm = _init_random(rng, rng.uniform(0.4, 0.7))
        rm = _repair(rm)
        # short SA
        m, c = _sa(rm, rng, t0=1.5, t1=0.01, iters=100000,
                   deadline=HARD_DEADLINE)
        if c > best_c:
            best_c = c
            best_mask = m

    # final safety: ensure feasibility and nonempty
    if not _is_200_free(best_mask) or best_mask == 0:
        best_mask = _init_interval(66, 100)
        best_mask = _repair(best_mask)

    # convert mask to sorted list of values
    out = []
    for v in range(1, N + 1):
        if (best_mask >> (v - 1)) & 1:
            out.append(v)
    if not out:
        out = list(range(66, 101))
    return out
