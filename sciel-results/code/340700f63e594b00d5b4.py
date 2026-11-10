"""Minimize distinct prime factors across pairwise sums of 12 distinct positive integers <= 10000.

Structural approach (arithmetic-progression / modular construction):
  - Pick a modulus m that is a product of a few small primes.
  - Choose 12 distinct integers forming an arithmetic progression with step m
    (so all pairwise sums are divisible by m, guaranteeing those primes appear).
  - Optimize the starting offset a (1-D) to minimize the number of ADDITIONAL
    distinct primes appearing in the sums a_i + a_j = 2a + m*(i+j).
  - Try many moduli and residue structures; keep the best.
  - Finish with a free-form local refinement.
"""
import time
import math
import random
from itertools import combinations

_START_TIME = time.time()
_DEADLINE_SECONDS = 82.0  # safety margin under 90s wall-clock


def _elapsed():
    return time.time() - _START_TIME


def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS


# ---------- sieve / prime factorization up to 20000 ----------
_MAX_SUM = 20000
_sieve = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _sieve[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _sieve[_j] == _j:
                _sieve[_j] = _i


def _prime_factors_fast(n):
    factors = set()
    while n > 1:
        p = _sieve[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors


_pf_cache = [None] * (_MAX_SUM + 1)


def _pf(s):
    r = _pf_cache[s]
    if r is None:
        r = _prime_factors_fast(s)
        _pf_cache[s] = r
    return r


N = 12
MAX_V = 10000


def _objective(nums):
    primes = set()
    for i in range(N):
        ai = nums[i]
        for j in range(i + 1, N):
            primes |= _pf(ai + nums[j])
    return len(primes)


def _objective_set(nums):
    primes = set()
    for i in range(N):
        ai = nums[i]
        for j in range(i + 1, N):
            primes |= _pf(ai + nums[j])
    return primes


def _valid(nums):
    if len(nums) != N:
        return False
    if len(set(nums)) != N:
        return False
    return all(1 <= x <= MAX_V for x in nums)


def _free_refine(nums, rng, iters, T0=20.0, T1=0.05):
    """Free-form local refinement minimizing distinct-prime count."""
    cur = nums[:]
    best = cur[:]
    best_obj = _objective(best)
    if best_obj <= 1:
        return best_obj, best

    def _delta_appear(idx, new_val):
        old_val = cur[idx]
        d = 0
        for j in range(N):
            if j == idx:
                continue
            d += len(_pf(new_val + cur[j])) - len(_pf(old_val + cur[j]))
        return d

    cur_cnt = sum(len(_pf(cur[i] + cur[j])) for i in range(N) for j in range(i + 1, N))
    best_cnt = cur_cnt

    for it in range(iters):
        if _deadline_hit():
            break
        T = T0 * (T1 / T0) ** (it / max(1, iters))
        idx = rng.randint(0, N - 1)
        if rng.random() < 0.75:
            new_val = cur[idx] + rng.randint(-300, 300)
        else:
            new_val = rng.randint(1, MAX_V)
        new_val = max(1, min(MAX_V, new_val))
        if new_val == cur[idx]:
            continue
        if any(cur[j] == new_val for j in range(N) if j != idx):
            continue
        d = _delta_appear(idx, new_val)
        if d < 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
            cur[idx] = new_val
            cur_cnt += d
            if cur_cnt < best_cnt:
                best_cnt = cur_cnt
                obj = _objective(cur)
                if obj < best_obj:
                    best_obj = obj
                    best = cur[:]
                    if best_obj <= 1:
                        break
    return best_obj, best


def _greedy_independent_set(conflict, target, rng):
    """Greedy: pick target vertices no two of which are in conflict.
    conflict is a set of frozenset pairs or a function. We pass a callable is_conflict(a,b)."""
    is_conflict = conflict
    chosen = []
    pool = list(range(1, MAX_V + 1))
    rng.shuffle(pool)
    for x in pool:
        if len(chosen) >= target:
            break
        if all(not is_conflict(x, y) for y in chosen):
            chosen.append(x)
    return chosen


def construct():
    rng = random.Random(20240517)
    best_score = 10 ** 9
    best_nums = list(range(1, 13))

    # small primes
    small_primes = []
    m = 2
    while len(small_primes) < 15:
        if _sieve[m] == m:
            small_primes.append(m)
        m += 1

    # ---------- Phase 1: arithmetic-progression / modular construction ----------
    # For a modulus M (product of small primes) and residue structure, choose 12 numbers
    # a + M*k (k in a fixed set of 12 distinct offsets) so all pairwise sums are divisible
    # by M. Then optimize the offset a over a 1-D range.
    #
    # We'll try several offset-sets K (12 distinct small nonnegative integers) and several M.

    # Candidate offset sets K (12 distinct values). Keep sums M*(i+j) within range by keeping
    # offsets small so that 2*a + M*max(i+j) <= MAX_V for some a>=1.
    offset_sets = [
        list(range(12)),
        list(range(0, 24, 2)),
        list(range(0, 36, 3)),
        list(range(1, 25, 2)),
        [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29],  # primes
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],  # powers of 2
    ]
    # add a couple random offset sets
    for _ in range(4):
        ks = sorted(rng.sample(range(0, 60), 12))
        offset_sets.append(ks)

    # Candidate moduli M: products of subsets of small primes (keep M <= ~2000 so 12 fit).
    moduli = []
    # single primes
    for p in small_primes[:8]:
        moduli.append(p)
    # products of 2
    for i in range(len(small_primes[:6])):
        for j in range(i + 1, len(small_primes[:6])):
            mm = small_primes[i] * small_primes[j]
            if mm <= 3000:
                moduli.append(mm)
    # products of 3 (small)
    for i in range(len(small_primes[:5])):
        for j in range(i + 1, len(small_primes[:5])):
            for k in range(j + 1, len(small_primes[:5])):
                mm = small_primes[i] * small_primes[j] * small_primes[k]
                if mm <= 3000:
                    moduli.append(mm)
    moduli = sorted(set(moduli))

    def _build_from_AP(a, M, K):
        nums = [a + M * k for k in K]
        return nums

    def _best_a_for(M, K):
        """Find offset a minimizing objective for nums = a + M*k, k in K."""
        max_k = max(K)
        # need a + M*max_k <= MAX_V and a >= 1
        a_max = MAX_V - M * max_k
        if a_max < 1:
            return None, None
        # objective depends on a via sums 2a + M*(i+j). We'll hill-climb a in [1, a_max].
        best_obj = 10 ** 9
        best_a = None
        # coarse scan
        step = max(1, a_max // 400)
        candidates = set()
        for a in range(1, a_max + 1, step):
            candidates.add(a)
        candidates.add(1)
        candidates.add(a_max)
        # local hill climb from a few seeds
        seeds = [1, a_max, a_max // 2]
        for a0 in seeds:
            a0 = max(1, min(a_max, a0))
            a = a0
            cur_obj = _objective(_build_from_AP(a, M, K))
            for _ in range(300):
                if _deadline_hit():
                    break
                improved = False
                for da in (1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 13, -13):
                    na = a + da
                    if na < 1 or na > a_max:
                        continue
                    o = _objective(_build_from_AP(na, M, K))
                    if o < cur_obj:
                        a = na
                        cur_obj = o
                        improved = True
                        break
                if not improved:
                    # random jump
                    a = rng.randint(1, a_max)
                    cur_obj = _objective(_build_from_AP(a, M, K))
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_a = a
        # also evaluate coarse-scan best
        for a in candidates:
            o = _objective(_build_from_AP(a, M, K))
            if o < best_obj:
                best_obj = o
                best_a = a
        return best_obj, best_a

    # Evaluate modular constructions
    for M in moduli:
        if _deadline_hit():
            break
        for K in offset_sets:
            if _deadline_hit():
                break
            max_k = max(K)
            if M * max_k >= MAX_V - 1:
                continue
            obj, a = _best_a_for(M, K)
            if a is None:
                continue
            nums = _build_from_AP(a, M, K)
            if not _valid(nums):
                continue
            if obj < best_score:
                best_score = obj
                best_nums = nums[:]
            if best_score <= 1:
                break
        if best_score <= 1:
            break

    # ---------- Phase 2: maximum-independent-set style on a conflict graph ----------
    # Vertices: integers 1..MAX_V. Edge between x,y if x+y introduces a prime outside a
    # chosen small set S. Greedily select 12 mutually non-adjacent vertices, then refine.
    # We approximate: pick S as the primes of current best; build conflict = introduces new prime.
    S_cur = _objective_set(best_nums)
    for S_size in (2, 3, 4, 5):
        if _deadline_hit():
            break
        # try several small prime sets S
        S_trials = [set(small_primes[:S_size]), set(small_primes[1:S_size + 1])]
        # also S = current best primes (if small)
        if 2 <= len(S_cur) <= S_size:
            S_trials.append(set(S_cur))
        for S in S_trials:
            if _deadline_hit():
                break

            def is_conflict(x, y, S=S):
                s = x + y
                for p in _pf(s):
                    if p not in S:
                        return True
                return False

            # Greedy independent set with multiple random orders
            found = False
            for _ in range(30):
                if _deadline_hit():
                    break
                chosen = _greedy_independent_set(is_conflict, N, rng)
                if len(chosen) == N:
                    obj = _objective(chosen)
                    if obj < best_score:
                        best_score = obj
                        best_nums = chosen[:]
                        found = True
                        if best_score <= 1:
                            break
            if best_score <= 1:
                break

    # ---------- Phase 3: free-form refinement ----------
    if not _deadline_hit():
        obj, nums = _free_refine(best_nums, rng, iters=60000, T0=25.0, T1=0.02)
        if obj < best_score:
            best_score = obj
            best_nums = nums[:]

    # Second free refine from a few diverse starts if time remains
    if not _deadline_hit() and best_score > 1:
        diverse = [
            list(range(1, 13)),
            [2 ** i for i in range(1, 13)],
            sorted(rng.sample(range(1, MAX_V + 1), N)),
            [6 * (i + 1) for i in range(12)],
        ]
        for start in diverse:
            if _deadline_hit():
                break
            if not _valid(start):
                continue
            obj, nums = _free_refine(start, rng, iters=20000, T0=30.0, T1=0.05)
            if obj < best_score:
                best_score = obj
                best_nums = nums[:]
                if best_score <= 1:
                    break

    result = best_nums
    assert _valid(result)
    return result


# EVOLVE-BLOCK-END
