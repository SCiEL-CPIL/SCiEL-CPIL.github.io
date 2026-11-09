"""Direct construction for 'Large sets with distinct subset products'.

Structural core (from the prime-power / per-prime sum-free framework):
  A set S has all 2^|S| subset products distinct iff, for every prime p, the
  multiset of p-exponents of the elements of S is "subset-sum-free" (no entry
  equals a subset-sum of the other entries).  This is a necessary condition
  and is also sufficient for prime-power constructions, so it is a sound, fast
  prune.  We maximise |S| over integers in [2, 40] using a greedy + local
  swap search guided by that prune, and finish with a full 2^k product
  enumeration as the final guarantee.
"""

import time
import itertools

DEADLINE = 82.0          # wall-clock budget (90s limit, safety margin)
T_START = time.time()

# primes in [2, 40]
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
ALL_CANDIDATES = list(range(2, 41))


def _exponent_of(n, p):
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


def _subset_sum_free(pos):
    """True iff no element of `pos` equals a subset-sum of the other elements."""
    m = len(pos)
    for i in range(m):
        target = pos[i]
        others = [pos[j] for j in range(m) if j != i]
        s = len(others)
        for mask in range(1, 1 << s):
            acc = 0
            mm = mask
            j = 0
            while mm:
                if mm & 1:
                    acc += others[j]
                    if acc > target:
                        break
                j += 1
                mm >>= 1
            if acc == target:
                return False
    return True


def _per_prime_ok(numbers, prime_idx=None):
    """Check the per-prime sum-free condition (necessary for distinct products)."""
    for pi, p in enumerate(PRIMES):
        if prime_idx is not None and pi != prime_idx:
            continue
        pos = []
        for n in numbers:
            e = _exponent_of(n, p)
            if e:
                pos.append(e)
        if not _subset_sum_free(pos):
            return False
    return True


def _is_valid(numbers):
    """Full guarantee: enumerate all 2^n subset products, check distinctness."""
    n = len(numbers)
    if n > 22:
        return None  # too large to enumerate; rely on the per-prime check
    prods = [1]
    for num in numbers:
        prods += [p * num for p in prods]
    if len(prods) != len(set(prods)):
        return False
    return True


def _build_prime_power_set():
    """Provable 16-element valid set: for each prime choose a sum-free exponent
    set (largest possible) and include the corresponding prime powers."""
    items = []
    used = set()
    for p in PRIMES:
        m = 1
        while p ** (m + 1) <= 40:
            m += 1
        exps = list(range(1, m + 1))
        best = []
        for k in range(len(exps), -1, -1):
            found = False
            for combo in itertools.combinations(exps, k):
                if _subset_sum_free(list(combo)):
                    best = list(combo)
                    found = True
                    break
            if found:
                break
        for e in best:
            v = p ** e
            if 2 <= v <= 40 and v not in used:
                items.append(v)
                used.add(v)
    return items


def _greedy_add(base, time_left):
    """Repeatedly try to add any candidate that keeps the per-prime condition."""
    cur = list(base)
    cur_set = set(cur)
    for _ in range(12):
        if time_left() < 8.0:
            break
        added = False
        for cand in ALL_CANDIDATES:
            if cand in cur_set:
                continue
            trial = cur + [cand]
            if _per_prime_ok(trial):
                cur = trial
                cur_set.add(cand)
                added = True
        if not added:
            break
    return cur


def _try_swap_extend(base, time_left):
    """Try single swaps; if a swap yields a valid set, extend it greedily."""
    cur = list(base)
    cur_set = set(cur)
    n = len(cur)
    for i in range(n):
        if time_left() < 6.0:
            break
        removed = cur[i]
        without = cur[:i] + cur[i + 1:]
        for cand in ALL_CANDIDATES:
            if cand in cur_set:
                continue
            trial = without + [cand]
            if not _per_prime_ok(trial):
                continue
            extended = _greedy_add(trial, time_left)
            if len(extended) > len(cur):
                cur = extended
                cur_set = set(cur)
                break
    return cur


def construct():
    global DEADLINE, T_START
    deadline = DEADLINE
    if deadline < 0:
        deadline = time.time() + 82.0

    def time_left():
        return deadline - time.time()

    # Strong provable baseline (16 elements) from the prime-power framework.
    base = _build_prime_power_set()
    if _is_valid(base) is not True:
        base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    best = _greedy_add(base, time_left)

    # Local search: swaps to unlock further additions.
    if time_left() > 25.0:
        improved = True
        while improved and time_left() > 15.0:
            improved = False
            cand_best = _try_swap_extend(best, time_left)
            if len(cand_best) > len(best):
                best = cand_best
                improved = True
            else:
                # one more greedy pass on the current best
                extended = _greedy_add(best, time_left)
                if len(extended) > len(best):
                    best = extended
                    improved = True

    # Final full verification; fall back to the guaranteed set if anything broke.
    if _is_valid(best) is not True:
        best = base if _is_valid(base) is True else \
            [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    res = [int(x) for x in best]
    assert all(isinstance(x, int) and not isinstance(x, bool) and 2 <= x <= 40
               for x in res)
    assert len(res) == len(set(res))
    return res
