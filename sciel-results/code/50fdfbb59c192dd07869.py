"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    # Known 7-term Egyptian fraction with denominators >= 5
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/70 = 1
    solution = [5, 6, 7, 10, 14, 35, 70]
    
    # Verify the solution
    s = sum(Fraction(1, d) for d in solution)
    if s == 1:
        return solution
    
    # Fallback: try to find a solution via DFS
    start = time.time()
    deadline = start + 55

    def dfs(remaining, min_d, max_terms, current, max_d_limit):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current
        if max_terms == 0:
            return None
        if remaining <= 0:
            return None

        # Pruning: max possible sum with max_terms terms starting from min_d
        max_possible = 0
        for i in range(max_terms):
            max_possible += Fraction(1, min_d + i)
        if remaining > max_possible:
            return None

        # Pruning: min possible sum
        min_possible = max_terms * Fraction(1, max_d_limit)
        if remaining < min_possible:
            return None

        # Bounds for next denominator d
        if remaining >= 1:
            lower = min_d
        else:
            lower = max(min_d, int(1 / float(remaining)) + 1)

        # Upper bound from min constraint: remaining - 1/d >= (max_terms-1)/max_d_limit
        thresh = remaining - (max_terms - 1) * Fraction(1, max_d_limit)
        if thresh <= 0:
            return None
        upper_from_min = int(1 / float(thresh))
        upper = min(max_d_limit, upper_from_min)

        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return None
            frac = Fraction(1, d)
            if frac > remaining:
                continue
            new_remaining = remaining - frac
            if new_remaining < 0:
                continue
            if max_terms - 1 == 0:
                if new_remaining == 0:
                    return current + [d]
                continue
            result = dfs(new_remaining, d + 1, max_terms - 1, current + [d], max_d_limit)
            if result is not None:
                return result
        return None

    # Search for minimum number of terms
    for k in range(2, 15):
        result = dfs(Fraction(1, 1), 5, k, [], 1000000)
        if result is not None:
            return result

    # Fallback to DP approach
    target_lcm = 2520
    reached = {0: []}
    for d in range(5, target_lcm + 1):
        if target_lcm % d:
            continue
        w = target_lcm // d
        for total, values in list(reached.items()):
            if total + w <= target_lcm and total + w not in reached:
                reached[total + w] = values + [d]
        if target_lcm in reached:
            return reached[target_lcm]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
