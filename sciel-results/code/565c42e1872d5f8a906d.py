"""Fast integer LCM-based reciprocal-subsum-free set maximizer with meet-in-the-middle
violation database and exhaustive local search."""
import time
import math
import random

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for _i in range(1, 25):
    _LCM = _lcm(_LCM, _i)

# v[i] = LCM / i, so sum of reciprocals = sum of v's / LCM
_V = [0] * 25
for _i in range(1, 25):
    _V[_i] = _LCM // _i

# Precompute subset-sum sets for every 12-bit mask (incremental DP, fast)
_NSUB = 1 << 12
_left = list(range(1, 13))
_right = list(range(13, 25))

_left_sums = [None] * _NSUB
_right_sums = [None] * _NSUB
_left_sums[0] = frozenset((0,))
_right_sums[0] = frozenset((0,))
for _m in range(1, _NSUB):
    _lsb = _m & -_m
    _i = _lsb.bit_length() - 1
    _prev = _m ^ _lsb
    _pv = _V[_left[_i]]
    _ps = _left_sums[_prev]
    _left_sums[_m] = frozenset(_ps | {x + _pv for x in _ps})
    _pv = _V[_right[_i]]
    _ps = _right_sums[_prev]
    _right_sums[_m] = frozenset(_ps | {x + _pv for x in _ps})

def _check_safe(i, S):
    """True iff element (i+1) is safe in set S (its reciprocal not a subsum of others)."""
    target = _V[i + 1]
    avail = S & ~(1 << i)
    if avail == 0:
        return True
    lm = avail & 0xFFF
    rm = (avail >> 12) & 0xFFF
    Ls = _left_sums[lm]
    Rs = _right_sums[rm]
    if len(Ls) < len(Rs):
        for sL in Ls:
            if (target - sL) in Rs:
                return False
    else:
        for sR in Rs:
            if (target - sR) in Ls:
                return False
    return True

def _is_valid(S):
    m = S
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        if not _check_safe(i, S):
            return False
        m ^= lsb
    return True

def _greedy(order, deadline):
    S = 0
    for idx in order:
        if time.time() > deadline:
            break
        trial = S | (1 << idx)
        ok = True
        m = trial
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if not _check_safe(i, trial):
                ok = False
                break
            m ^= lsb
        if ok:
            S = trial
    return S

def _local_search(S, deadline, max_iters=100):
    best = S
    best_size = best.bit_count()
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        # add one
        for i in range(24):
            if time.time() > deadline:
                break
            if not (best >> i) & 1:
                trial = best | (1 << i)
                if _is_valid(trial):
                    best = trial
                    best_size += 1
                    improved = True
                    break
        if improved:
            continue
        # swap one for one
        for i in range(24):
            if time.time() > deadline:
                break
            if (best >> i) & 1:
                base = best & ~(1 << i)
                for j in range(24):
                    if time.time() > deadline:
                        break
                    if not (base >> j) & 1:
                        trial = base | (1 << j)
                        if _is_valid(trial):
                            best = trial
                            improved = True
                            break
                if improved:
                    break
        if improved:
            continue
        # remove one, add two
        for i in range(24):
            if time.time() > deadline:
                break
            if (best >> i) & 1:
                base = best & ~(1 << i)
                for j in range(24):
                    if time.time() > deadline:
                        break
                    if not (base >> j) & 1:
                        for k in range(j + 1, 24):
                            if time.time() > deadline:
                                break
                            if not (base >> k) & 1:
                                trial = base | (1 << j) | (1 << k)
                                if trial.bit_count() > best_size and _is_valid(trial):
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
        # remove two, add three
        if not improved:
            for i in range(24):
                if time.time() > deadline:
                    break
                if (best >> i) & 1:
                    for j in range(i + 1, 24):
                        if time.time() > deadline:
                            break
                        if (best >> j) & 1:
                            base = best & ~(1 << i) & ~(1 << j)
                            for k in range(24):
                                if time.time() > deadline:
                                    break
                                if not (base >> k) & 1:
                                    for l in range(k + 1, 24):
                                        if time.time() > deadline:
                                            break
                                        if not (base >> l) & 1:
                                            for m2 in range(l + 1, 24):
                                                if time.time() > deadline:
                                                    break
                                                if not (base >> m2) & 1:
                                                    trial = base | (1 << k) | (1 << l) | (1 << m2)
                                                    if trial.bit_count() > best_size and _is_valid(trial):
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
    start = time.time()
    deadline = start + 85.0
    best_set = 0
    best_size = 0

    def update(S):
        nonlocal best_set, best_size
        sz = S.bit_count()
        if sz > best_size:
            best_set = S
            best_size = sz

    # Element 1 is always safe: 1 = sum of reciprocals of distinct ints >= 2 is impossible.
    # So we always include element 1 (index 0).
    base = 1  # bit 0 set

    orders = [
        list(range(23, 0, -1)),   # 24..2 descending
        list(range(1, 24)),       # 2..24 ascending
        [23, 12, 22, 11, 21, 10, 20, 9, 19, 8, 18, 7, 17, 6, 16, 5, 15, 4, 14, 3, 13, 2],
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2],
        # Additional orders
        [2, 23, 3, 22, 4, 21, 5, 20, 6, 19, 7, 18, 8, 17, 9, 16, 10, 15, 11, 14, 12, 13],
        [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    for order in orders:
        if time.time() > deadline:
            break
        S = base | _greedy(order, deadline)
        update(S)
        if S.bit_count() >= best_size - 1:
            S2 = _local_search(S, deadline, max_iters=80)
            update(S2)

    # Random restarts with local search
    random.seed(12345)
    for attempt in range(6000):
        if time.time() > deadline:
            break
        elems = list(range(1, 24))
        random.shuffle(elems)
        S = base | _greedy(elems, deadline)
        if S.bit_count() >= best_size - 2:
            S2 = _local_search(S, deadline, max_iters=60)
            update(S2)

    # Full-set repair: start from all elements, remove violators
    if time.time() < deadline - 10:
        S = (1 << 24) - 1
        changed = True
        while changed and time.time() < deadline:
            changed = False
            m = S
            while m:
                if time.time() > deadline:
                    break
                lsb = m & -m
                i = lsb.bit_length() - 1
                if not _check_safe(i, S):
                    S &= ~(1 << i)
                    changed = True
                    break
                m ^= lsb
        update(S)
        S2 = _local_search(S, deadline, max_iters=100)
        update(S2)

    # Additional: try to find sets by starting from random subsets of size near best
    if time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            # Start with a random subset of size best_size - 2 to best_size
            sz = max(5, best_size - 2 + random.randint(0, 2))
            chosen = random.sample(range(1, 24), min(sz, 23))
            S = base
            for i in chosen:
                trial = S | (1 << i)
                if _is_valid(trial):
                    S = trial
            if S.bit_count() >= best_size - 1:
                S2 = _local_search(S, deadline, max_iters=80)
                update(S2)

    if best_set == 0:
        best_set = 1

    result = [i + 1 for i in range(24) if (best_set >> i) & 1]
    if not result:
        result = [1]
    return result

if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Result: {r}, size={len(r)}, time={t1-t0:.2f}s")
