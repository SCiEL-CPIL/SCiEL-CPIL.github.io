import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin for 60s limit

    # Use integer arithmetic: represent remaining fraction as (num, den)
    # where num/den is the remaining sum to achieve.

    def dfs(num, den, min_d, k_left, path):
        if time.time() > deadline:
            return None

        if k_left == 0:
            if num == 0:
                return path[:]
            return None

        if k_left == 1:
            # 1/d = num/den => d = den/num
            if num != 0 and den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000:
                    return path + [d]
            return None

        # Lower bound for d: 1/d <= num/den => d >= den/num
        # d must be at least min_d
        d_min = max(min_d, den // num + (1 if den % num else 0))
        # Ensure 1/d <= num/den, i.e., d >= den/num
        # If den/num is exact, d_min = den/num works since 1/d = num/den
        # Actually we need 1/d <= num/den => d >= den/num
        # So d_min = max(min_d, ceil(den/num))
        if den % num == 0:
            d_min_candidate = den // num
        else:
            d_min_candidate = den // num + 1
        d_min = max(min_d, d_min_candidate)

        # Upper bound: we need to form (num/den - 1/d) with k_left-1 terms from d+1 onwards
        # The max sum of k_left-1 terms starting from d+1 is approximately (k_left-1)/(d+1)
        # So num/den - 1/d <= (k_left-1)/(d+1)
        # This gives an upper bound on d.
        # Also, the max possible sum with k_left terms from min_d:
        # sum_{i=0}^{k_left-1} 1/(min_d+i) must be >= num/den
        # Quick float check:
        max_possible = 0.0
        for i in range(k_left):
            max_possible += 1.0 / (min_d + i)
        if max_possible < num / den - 1e-12:
            return None

        # Upper bound for d:
        # We need num/den - 1/d <= sum_{j=0}^{k_left-2} 1/(d+1+j)
        # Approx: num/den - 1/d <= (k_left-1)/(d+1)
        # num/den <= 1/d + (k_left-1)/(d+1)
        # For large d, RHS ~ k_left/d, so d <= k_left * den / num
        d_max = (k_left * den) // num + 2
        if d_max < d_min:
            return None
        # Cap d_max to avoid too large ranges
        if d_max > 10000000:
            d_max = 10000000

        for d in range(d_min, d_max + 1):
            # Check 1/d <= num/den => den <= num*d
            if den > num * d:
                continue

            # new_rem = num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d

            # Reduce fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g

            if new_num < 0:
                break  # 1/d > num/den, shouldn't happen given our check

            # Pruning: can we form new_num/new_den with k_left-1 terms from d+1?
            # Max sum of k_left-1 terms starting from d+1:
            max_next = 0.0
            for i in range(k_left - 1):
                max_next += 1.0 / (d + 1 + i)
            if new_num / new_den > max_next + 1e-12:
                break

            path.append(d)
            res = dfs(new_num, new_den, d + 1, k_left - 1, path)
            if res:
                return res
            path.pop()

        return None

    # Try small k first
    for k in range(7, 14):
        if time.time() > deadline - 3.0:
            break
        sol = dfs(1, 1, 5, k, [])
        if sol:
            return sol

    # Fallback: known solution with more terms
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16
    # This won't sum to 1 exactly, so use DP fallback
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
        if time.time() > deadline:
            break

    if target in reached:
        return reached[target]
    raise RuntimeError("no baseline representation")
