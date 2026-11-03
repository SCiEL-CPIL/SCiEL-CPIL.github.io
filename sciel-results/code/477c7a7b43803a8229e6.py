"""Twin-prime ladder construction with time-budgeted search and polish."""
import time
import math
from bisect import bisect_left

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # 10s safety margin under the 150s kill


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    d = 5
    w = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += w
        w = 6 - w
    return True


def _score_set(U):
    """Lower bound 1 + log(|U-U|/|U+U|)/log(2*maxU+1); 0 if invalid."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    mx = max(U)
    if mx <= 0:
        return 0.0
    su = set()
    du = set()
    for a in U:
        for b in U:
            su.add(a + b)
            du.add(a - b)
    lu = len(su)
    ld = len(du)
    if ld <= 0 or lu <= 0:
        return 0.0
    denom = math.log(2 * mx + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ld / lu) / denom


def _twin_ladder(start, k, scale=1):
    """U = cumulative sums of first k twin-prime pairs starting at >= start,
    with each gap multiplied by scale."""
    U = [0]
    cur = 0
    p = start
    count = 0
    while count < k:
        if _is_prime(p) and _is_prime(p + 2):
            cur += scale * (p + (p + 2))
            U.append(cur)
            count += 1
            p += 2
        else:
            p += 2
    return U


def _polish(U, max_rounds=4000):
    """Local search: try adding a new element or removing a non-zero element
    if it improves the score."""
    U = sorted(set(U))
    best = _score_set(U)
    if best <= 0.0:
        return U
    n = len(U)
    mx = max(U)
    # candidate additions: a few values around gaps and near 2*mx
    cand_add = set()
    for a in U:
        cand_add.add(a + 1)
        cand_add.add(a + 2)
    cand_add.add(mx + 1)
    cand_add.add(mx + 2)
    cand_add.add(mx + 3)
    cand_add.add(2 * mx)
    cand_add.add(2 * mx + 1)
    Uset = set(U)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < _DEADLINE:
        improved = False
        rounds += 1
        # try removals (never remove 0)
        for a in sorted(U):
            if a == 0:
                continue
            if len(U) <= 3:
                continue
            trial = [x for x in U if x != a]
            s = _score_set(trial)
            if s > best + 1e-12:
                best = s
                U = trial
                Uset = set(U)
                improved = True
                break
        if improved:
            continue
        # try additions
        for a in sorted(cand_add):
            if a in Uset or a < 0:
                continue
            trial = U + [a]
            s = _score_set(trial)
            if s > best + 1e-12:
                best = s
                U = sorted(trial)
                Uset = set(U)
                mx = max(U)
                cand_add.add(a + 1)
                cand_add.add(a + 2)
                cand_add.add(mx + 1)
                cand_add.add(mx + 2)
                improved = True
                break
    return U


def construct():
    t0 = time.time()
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)

    # Twin-prime ladder search: grid over start, k, scale
    starts = [3, 5, 11, 17, 29, 41, 59, 71, 101, 137, 173, 239, 347, 419, 523, 719]
    ks = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 80, 130, 200, 320, 500]
    scales = [1, 2, 3, 5, 8, 13]

    # Precompute twin primes up to a bound for fast ladder building
    # We'll just build on the fly but cache primes.
    prime_cache = {}

    def prime_cached(n):
        if n not in prime_cache:
            prime_cache[n] = _is_prime(n)
        return prime_cache[n]

    def twin_ladder_cached(start, k, scale):
        U = [0]
        cur = 0
        p = start
        count = 0
        while count < k:
            if prime_cached(p) and prime_cached(p + 2):
                cur += scale * (2 * p + 2)
                U.append(cur)
                count += 1
                p += 2
            else:
                p += 2
        return U

    # Evaluate ladders, keep best
    for start in starts:
        for k in ks:
            for scale in scales:
                if time.time() > _DEADLINE - 20:
                    break
                U = twin_ladder_cached(start, k, scale)
                if len(U) < 2:
                    continue
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    # Also try a few hand-tuned small sets and known good structures
    extra_sets = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 4, 6, 7, 9],
        [0, 1, 3, 4, 6, 9, 10, 12],
    ]
    for U in extra_sets:
        if time.time() > _DEADLINE - 15:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = U

    # Polish the best set found
    if time.time() < _DEADLINE - 5:
        polished = _polish(list(best_set), max_rounds=2000)
        s = _score_set(polished)
        if s > best_score:
            best_score = s
            best_set = polished

    # Final polish with more budget if time remains
    if time.time() < _DEADLINE - 3:
        polished2 = _polish(list(best_set), max_rounds=3000)
        s = _score_set(polished2)
        if s > best_score:
            best_score = s
            best_set = polished2

    return sorted(set(best_set))
