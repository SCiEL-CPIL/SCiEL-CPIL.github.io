import time
import math
from itertools import combinations

MAXV = 10000
N = 12
DEADLINE = 55.0  # safety margin
START = time.time()


def sieve(limit):
    is_p = bytearray(b'\x01') * (limit + 1)
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
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.add(d)
            n //= d
        d += 1
    if n > 1:
        out.add(n)
    return out


def cost_of(arr):
    s = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s |= factor(arr[i] + arr[j])
    return len(s), s


def smooth_values(primes):
    """Generate all numbers <= MAXV whose prime factors are all in primes."""
    ps = sorted(primes)
    vals = [1]
    for p in ps:
        new_vals = []
        for v in vals:
            x = v * p
            while x <= MAXV:
                new_vals.append(x)
                x *= p
        vals = list(set(vals + new_vals))
    return sorted(vals), set(vals)


def try_smooth_set(primes, deadline):
    """Try to find N distinct values from smooth values such that all pairwise sums are smooth."""
    vals, vset = smooth_values(primes)
    
    # We need 12 distinct values such that every pairwise sum is in vset
    # This is equivalent to: for all i != j, vals[i] + vals[j] in vset
    
    # Greedy + backtracking
    best = None
    best_cost = 10 ** 9
    
    def rec(chosen, start_idx):
        nonlocal best, best_cost
        if time.time() > deadline:
            return
        if len(chosen) == N:
            best = list(chosen)
            best_cost = len(primes)
            return
        
        for k in range(start_idx, len(vals)):
            x = vals[k]
            ok = True
            for c in chosen:
                if (x + c) not in vset:
                    ok = False
                    break
            if not ok:
                continue
            rec(chosen + [x], k + 1)
            if best is not None:
                return
    
    rec([], 0)
    return best, best_cost


def construct():
    deadline = START + DEADLINE - 1.0  # extra safety margin

    # 1) Precomputed strong candidate (12 numbers, all pairwise sums 2,3-smooth)
    pre = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    c, _ = cost_of(pre)
    best = pre
    best_c = c  # expect 2

    if time.time() > deadline:
        return best

    # 2) Try 2-prime sets to confirm or find better
    # Start with {2,3} which we know works
    # Then try other pairs

    # Try {2, p} for small primes p
    two_sets = [(2, 3)]  # known to work
    for p in PRIMES:
        if p > 100:
            break
        if p != 2:
            two_sets.append((2, p))
        if p != 3:
            two_sets.append((3, p))
    two_sets = list(set(two_sets))
    two_sets.sort(key=lambda t: (min(t), max(t)))

    for ps in two_sets:
        if time.time() > deadline:
            break
        sol, cc = try_smooth_set(list(ps), deadline)
        if sol is not None:
            # Verify
            c2, _ = cost_of(sol)
            if c2 < best_c:
                best, best_c = sol, c2
            if best_c == 2:
                # Can't do better than 2 distinct primes
                pass

    # 3) If we haven't found a 2-prime solution, try 3-prime sets
    if best_c > 2 and time.time() < deadline:
        three_sets = list(combinations(PRIMES[:30], 3))
        three_sets.sort(key=lambda t: min(t))
        for ps in three_sets[:100]:  # limit to 100 combinations
            if time.time() > deadline:
                break
            sol, cc = try_smooth_set(list(ps), deadline)
            if sol is not None:
                c2, _ = cost_of(sol)
                if c2 < best_c:
                    best, best_c = sol, c2
                if best_c == 2:
                    break

    # 4) Local search improvement
    if time.time() < deadline - 2.0 and best_c > 2:
        arr = list(best)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            cur_c, _ = cost_of(arr)
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                # Try replacing arr[i] with other values
                for v in range(1, MAXV + 1, 10):  # sample every 10th value
                    if v in arr:
                        continue
                    old = arr[i]
                    arr[i] = v
                    nc, _ = cost_of(arr)
                    if nc < cur_c:
                        cur_c = nc
                        improved = True
                        break
                    else:
                        arr[i] = old
                if time.time() > deadline - 2.0:
                    break
            if cur_c < best_c:
                best_c = cur_c
                best = list(arr)

    return list(best)
