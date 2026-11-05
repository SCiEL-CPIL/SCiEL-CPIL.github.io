"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85  # 85s safety margin (90s limit)
    MAX_DEN = 10**7

    def dfs(rem_num, rem_den, min_d, terms_left, path, used_set):
        """
        rem_num/rem_den: fraction remaining to reach 1
        min_d: next denominator must be >= min_d
        terms_left: how many more terms to use
        path: list of chosen denominators so far
        used_set: set of used denominators
        Returns list of denominators or None
        """
        if time.time() > deadline:
            return None
        if rem_num == 0:
            return path
        if terms_left == 0:
            return None

        # Pruning: max possible sum with terms_left terms starting from min_d
        # Each term <= 1/min_d, so sum <= terms_left / min_d
        # Need rem_num/rem_den <= terms_left / min_d
        # => rem_num * min_d <= rem_den * terms_left
        if rem_num * min_d > rem_den * terms_left:
            return None

        # Lower bound on next denominator: 1/d <= rem => d >= rem_den/rem_num
        den_lb = max(min_d, (rem_den + rem_num - 1) // rem_num)

        if den_lb > MAX_DEN:
            return None

        # Upper bound on next denominator: 1/d >= rem/terms_left => d <= terms_left*rem_den/rem_num
        den_ub = (terms_left * rem_den) // rem_num
        if den_ub > MAX_DEN:
            den_ub = MAX_DEN
        if den_ub < den_lb:
            return None

        # Cap search range for efficiency - try a window
        # For the first few terms, the range might be large; cap it
        search_end = min(den_ub, den_lb + 200000)

        for d in range(den_lb, search_end + 1):
            if time.time() > deadline:
                return None
            if d in used_set:
                continue
            # 1/d must be <= rem: d >= rem_den/rem_num (already ensured by den_lb)
            # Compute new_rem = rem - 1/d = (rem_num*d - rem_den) / (rem_den*d)
            new_num = rem_num * d - rem_den
            if new_num < 0:
                continue
            new_den = rem_den * d
            # Reduce fraction
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g

            if new_num == 0:
                if terms_left == 1:
                    return path + [d]
                else:
                    continue  # need exactly terms_left terms, can't stop early

            # Pruning for remaining: new_rem must be achievable with terms_left-1 terms
            # Max sum: (terms_left-1) / (d+1)
            if terms_left - 1 > 0:
                if new_num * (d + 1) > new_den * (terms_left - 1):
                    continue

            result = dfs(new_num, new_den, d + 1, terms_left - 1, path + [d], used_set | {d})
            if result is not None:
                return result
        return None

    # k=9 is the minimum feasible (k<9 impossible since max sum < 1)
    result = dfs(1, 1, 5, 9, [], frozenset())
    if result is not None:
        # Verify
        s_num = 0
        s_den = 1
        for d in result:
            g = gcd(s_den, d)
            s_num = s_num * (d // g) + s_den // g
            s_den = s_den * (d // g)
        g = gcd(s_num, s_den)
        s_num //= g
        s_den //= g
        if s_num == 1 and s_den == 1 and len(set(result)) == len(result) and all(5 <= d <= MAX_DEN for d in result):
            return result

    # Fallback: DP approach with common denominator
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no valid representation found")

# EVOLVE-BLOCK-END
