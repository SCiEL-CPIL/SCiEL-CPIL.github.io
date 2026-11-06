"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt

def construct():
    deadline = time.time() + 84.0  # safety margin before 90s kill
    target = factorial(12)          # 479001600 = 2^10 * 3^5 * 5^2 * 7

    # Prime exponent vector of 12!
    def exponents(n):
        e = [0, 0, 0, 0]
        for p, idx in ((2, 0), (3, 1), (5, 2), (7, 3)):
            while n % p == 0:
                n //= p
                e[idx] += 1
        return e

    full = exponents(target)         # [10, 5, 2, 1]

    # All divisors of 12! (396 total)
    divs = set()
    for i in range(1, isqrt(target) + 1):
        if target % i == 0:
            divs.add(i)
            divs.add(target // i)
    divs = sorted(d for d in divs if d > 12)

    # Precompute exponent vector + log for each valid divisor
    div_exp = [exponents(d) for d in divs]
    import math
    div_log = [math.log(d) for d in divs]
    log_target = math.log(target)

    def time_left():
        return deadline - time.time()

    def feasible(M, k):
        """Is there a subset of divs (all <= M, distinct, product=target) of size exactly k?"""
        if time_left() <= 0.5:
            return False
        # Feasibility: product of k smallest allowed factors must be <= target
        if k < 1:
            return False
        if k > 12:
            return False
        # smallest k distinct factors > 12 and <= M: 13..(12+k) but capped by M
        # quick lower bound on product
        lb = 1
        for j in range(1, k + 1):
            f = 12 + j
            if f > M:
                return False
            lb *= f
            if lb > target:
                return False
        # also each factor <= M so product <= M^k must be >= target
        if M ** k < target:
            return False

        # Meet-in-the-middle over the divisors <= M
        idxs = [i for i, d in enumerate(divs) if d <= M]
        n = len(idxs)
        if n < k:
            return False
        mid = n // 2
        left = idxs[:mid]
        right = idxs[mid:]

        def enum(part):
            """Yield (tuple(exponents), count) for each subset of part with product <= target."""
            m = len(part)
            results = []
            for mask in range(1 << m):
                if bin(mask).count('1') > k:
                    continue
                e = [0, 0, 0, 0]
                prod = 1
                cnt = 0
                ok = True
                for b in range(m):
                    if mask & (1 << b):
                        d = divs[part[b]]
                        prod *= d
                        if prod > target:
                            ok = False
                            break
                        cnt += 1
                        e[0] += div_exp[part[b]][0]
                        e[1] += div_exp[part[b]][1]
                        e[2] += div_exp[part[b]][2]
                        e[3] += div_exp[part[b]][3]
                if not ok:
                    continue
                results.append((tuple(e), cnt))
            return results

        L = enum(left)
        # index right results by exponent vector
        R = {}
        for e, c in enum(right):
            R[e] = c

        # For each left vector, need a right vector with exponents full - left
        for e, c in L:
            need = (full[0] - e[0], full[1] - e[1], full[2] - e[2], full[3] - e[3])
            if need[0] < 0 or need[1] < 0 or need[2] < 0 or need[3] < 0:
                continue
            if need in R:
                if c + R[need] == k:
                    return True
        return False

    # Binary search on M for the smallest feasible max factor
    lo, hi = 13, max(divs)
    best = None
    # First ensure a feasible M exists (single factor always works: M=target, k=1)
    # Binary search
    while lo <= hi and time_left() > 2.0:
        mid = (lo + hi) // 2
        # check if any k in 1..12 is feasible at this M
        found = False
        for k in range(1, 13):
            if feasible(mid, k):
                found = True
                break
        if found:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if best is None:
        return [target]

    # Now recover an actual decomposition at M = best
    # Use DFS with the tight bound best to find explicit factors
    best_factors = None

    def dfs(start_idx, remaining, chosen, current_max, count_left):
        nonlocal best_factors
        if time_left() <= 0.5:
            return True
        if remaining == 1:
            if 1 <= len(chosen) <= 12:
                best_factors = list(chosen)
            return True
        if count_left == 0:
            return False
        # Prune: current_max already exceeds best -> but we're bounding by best so ok
        # Need count_left factors all > 12, distinct, > last chosen, <= best
        last = chosen[-1] if chosen else 12
        # lower bound product
        lb = 1
        for j in range(1, count_left + 1):
            f = last + j
            if f > best:
                return False
            lb *= f
            if lb > remaining:
                return False
        if best ** count_left < remaining:
            return False

        for i in range(start_idx, len(divs)):
            d = divs[i]
            if d > best:
                break
            if d <= last:
                continue
            if remaining % d != 0:
                continue
            if d > remaining:
                break
            chosen.append(d)
            if dfs(i + 1, remaining // d, chosen, max(current_max, d), count_left - 1):
                return True
            chosen.pop()
        return False

    # Try to find decomposition with the tight bound
    for k in range(1, 13):
        if time_left() <= 1.0:
            break
        if dfs(0, target, [], 0, k):
            break

    if best_factors is None:
        # fallback: single factor
        return [target]
    return best_factors
# EVOLVE-BLOCK-END
