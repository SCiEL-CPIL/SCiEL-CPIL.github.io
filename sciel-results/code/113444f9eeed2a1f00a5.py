import time
import math
from math import factorial

def construct():
    deadline = time.time() + 55.0
    N = factorial(12)  # 479001600

    # divisors of N = 2^10 * 3^5 * 5^2 * 7 * 11
    primes_exp = [(2,10),(3,5),(5,2),(7,1),(11,1)]
    divisors = [1]
    for p, e in primes_exp:
        new = []
        pw = 1
        for _ in range(e+1):
            for d in divisors:
                new.append(d*pw)
            pw *= p
        divisors = new
    divisors = sorted(set(d for d in divisors if d > 12))
    nd = len(divisors)

    best = [None]
    best_max = [N]

    # greedy init
    def greedy():
        cur = []
        rem = N
        last = 12
        while rem > 1:
            found = False
            for f in divisors:
                if f <= last:
                    continue
                if rem % f == 0:
                    cur.append(f)
                    rem //= f
                    last = f
                    found = True
                    break
            if not found:
                return None
        if rem == 1 and len(cur) <= 12:
            return cur
        return None
    g = greedy()
    if g is not None:
        best[0] = g
        best_max[0] = max(g)

    import sys
    sys.setrecursionlimit(100000)

    log_best_max = math.log(best_max[0]) if best_max[0] > 1 else 0
    nodes = [0]

    def dfs(idx, remaining, cur):
        if time.time() > deadline:
            return
        nodes[0] += 1
        if remaining == 1:
            if len(cur) <= 12:
                m = max(cur)
                if m < best_max[0]:
                    best_max[0] = m
                    best[0] = list(cur)
                    nonlocal log_best_max
                    log_best_max = math.log(m) if m > 1 else 0
            return
        # remaining_slots after this pick
        for i in range(idx, nd):
            f = divisors[i]
            if f > best_max[0]:
                break
            if f <= (cur[-1] if cur else 12):
                continue
            if len(cur) >= 12:
                break
            if remaining % f != 0:
                continue
            R = remaining // f
            # prune: R must be factorable into factors > f, <= best_max
            if R > 1 and R <= f:
                continue
            slots = 12 - len(cur) - 1
            if slots < 0:
                break
            if R > 1 and math.log(R) > slots * log_best_max + 1e-9:
                continue
            cur.append(f)
            dfs(i+1, R, cur)
            cur.pop()

    dfs(0, N, [])

    if best[0] is not None:
        return best[0]
    return [N]
