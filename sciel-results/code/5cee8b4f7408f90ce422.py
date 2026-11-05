"""Few prime factors across pairwise sums: pick a prime set S and 12 distinct
numbers <=10000 such that every pairwise sum is S-smooth (only primes in S)."""
import time
from math import prod
from itertools import combinations
from bisect import bisect_left

MAXV = 10000
N = 12
DEADLINE = 60.0
START = time.time()


def sieve(limit):
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            step = i
            start = i * i
            is_p[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))
    return [i for i in range(2, limit + 1) if is_p[i]]


PRIMES = sieve(MAXV)
PSET = set(PRIMES)


def factor(n):
    out = set()
    for p in PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            out.add(p)
            n //= p
    if n > 1:
        out.add(n)
    return out


def cost_of(arr):
    s = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s |= factor(arr[i] + arr[j])
    return len(s), s


def cost_of_fast(arr, prime_set):
    """Check if all pairwise sums are smooth with respect to prime_set.
    Returns (num_extra_primes, extra_primes_set) where extra primes are those
    not in prime_set."""
    ps = set(prime_set)
    extra = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = factor(arr[i] + arr[j])
            extra |= (s - ps)
    return len(extra), extra


def smooth_set(primes):
    ps = list(primes)
    ps.sort()
    m = len(ps)
    vals = [1]
    for p in ps:
        cur = list(vals)
        v = p
        while v <= MAXV:
            cur.append(v)
            v *= p
        vals += cur
    vals = sorted(set(vals))
    return vals, set(vals)


def build_solution(primes, deadline):
    vals, vset = smooth_set(primes)
    maxs = [MAXV - v for v in vals]
    maxs.sort(reverse=True)

    # candidate values whose double is smooth (so x+x is fine if ever needed)
    cand = [v for v in vals if (2 * v) in vset]
    cand.sort()

    def feasible(x, chosen):
        for c in chosen:
            if (x + c) not in vset:
                return False
        return True

    best = None
    best_cost = 10 ** 9

    def rec(chosen, idx):
        nonlocal best, best_cost
        if time.time() > deadline:
            return
        if len(chosen) == N:
            best = list(chosen)
            best_cost = len(primes)
            return
        # upper bound pruning: remaining needed
        remaining = N - len(chosen)
        # candidate pool: values > last chosen (distinct, increasing)
        start = chosen[-1] + 1 if chosen else 1
        pos = bisect_left(cand, start)
        for k in range(pos, len(cand)):
            x = cand[k]
            if not feasible(x, chosen):
                continue
            rec(chosen + [x], k + 1)
            if best is not None:
                return

    rec([], 0)
    return best, best_cost


def search_prime_set(primes, deadline):
    # quick: build smooth set and try greedy/backtrack
    sol, _ = build_solution(primes, deadline)
    if sol is None:
        return None
    return sol


def try_prime_set(primes, deadline):
    sol = search_prime_set(primes, deadline)
    if sol is None:
        return None
    # verify
    c, _ = cost_of(sol)
    if c <= len(primes):
        return sol, c
    return None


def construct():
    deadline = START + DEADLINE - 2.0  # safety margin

    # 1) Try a known good solution: powers of 2 give even sums
    #    All pairwise sums of powers of 2 are even, so at least prime 2 divides them
    #    But other primes may appear. Let's test.
    pre = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    c, _ = cost_of(pre)
    best = pre
    best_c = c

    # 2) Try another candidate: numbers of form 2^a * 3^b
    #    We want pairwise sums to be 2,3-smooth
    #    Let's try a greedy construction
    def greedy_23_smooth():
        """Find 12 distinct numbers <= 10000 such that all pairwise sums are 2,3-smooth."""
        smooth_23 = set()
        a = 0
        while 2**a <= MAXV:
            b = 0
            while 2**a * 3**b <= MAXV:
                smooth_23.add(2**a * 3**b)
                b += 1
            a += 1
        
        # Try to find 12 numbers where all pairwise sums are in smooth_23
        # Use backtracking
        vals = sorted(smooth_23)
        
        def backtrack(chosen, start_idx):
            if len(chosen) == N:
                return chosen[:]
            for i in range(start_idx, len(vals)):
                x = vals[i]
                ok = True
                for c in chosen:
                    if (x + c) not in smooth_23:
                        ok = False
                        break
                if ok:
                    result = backtrack(chosen + [x], i + 1)
                    if result is not None:
                        return result
            return None
        
        return backtrack([], 0)
    
    sol_23 = greedy_23_smooth()
    if sol_23 is not None:
        c23, _ = cost_of(sol_23)
        if c23 < best_c:
            best = sol_23
            best_c = c23

    if time.time() > deadline:
        return list(best)

    # 3) Try 2-prime sets systematically
    # Focus on {2,3} first since it's most likely to work
    for ps in [(2, 3), (2, 5), (3, 5), (2, 7), (3, 7), (5, 7)]:
        if time.time() > deadline:
            break
        res = try_prime_set(list(ps), deadline)
        if res is not None:
            sol, cc = res
            if cc < best_c:
                best, best_c = sol, cc
            if best_c <= 2:
                break

    # 4) If still > 2, try more 2-prime sets
    if best_c > 2:
        two_sets = [(2, p) for p in PRIMES if p != 2 and p <= 500]
        two_sets += [(3, p) for p in PRIMES if p != 3 and p <= 500]
        two_sets = list(set(two_sets))
        two_sets.sort(key=lambda t: min(t))
        for ps in two_sets:
            if time.time() > deadline:
                break
            if ps in [(2, 3), (2, 5), (3, 5), (2, 7), (3, 7), (5, 7)]:
                continue  # already tried
            res = try_prime_set(list(ps), deadline)
            if res is not None:
                sol, cc = res
                if cc < best_c:
                    best, best_c = sol, cc
                if best_c <= 2:
                    break

    # 5) Local search: try to reduce prime count by swapping elements
    if time.time() < deadline - 2.0:
        arr = list(best)
        cur_c, _ = cost_of(arr)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                # Try replacing arr[i] with values from smooth sets
                # First try small values
                for v in range(1, 200):
                    if v == arr[i]:
                        continue
                    if v in arr:
                        continue
                    arr[i] = v
                    nc, _ = cost_of(arr)
                    if nc < cur_c:
                        cur_c = nc
                        improved = True
                        break
                    else:
                        arr[i] = best[i]
                if cur_c < best_c:
                    best_c = cur_c
                    best = list(arr)

    return list(best)
