import time
from fractions import Fraction

def construct():
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
        # Use exact Fraction for correctness
        max_possible = Fraction(0)
        for i in range(max_terms):
            max_possible += Fraction(1, min_d + i)
        if remaining > max_possible:
            return None

        # Pruning: min possible sum (all terms at max_d_limit)
        min_possible = max_terms * Fraction(1, max_d_limit)
        if remaining < min_possible:
            return None

        # Bounds for next denominator d
        # We need 1/d <= remaining, so d >= ceil(1/remaining)
        # But also d >= min_d
        if remaining >= 1:
            lower = min_d
        else:
            # d must satisfy 1/d <= remaining, i.e., d >= 1/remaining
            # Use Fraction to avoid floating point issues
            inv_rem = Fraction(1, 1) / remaining
            # d >= ceil(1/remaining)
            lower = max(min_d, (inv_rem.numerator + inv_rem.denominator - 1) // inv_rem.denominator)
        
        # Upper bound: we need remaining - 1/d >= (max_terms - 1) / max_d_limit
        # So 1/d <= remaining - (max_terms-1)/max_d_limit
        # d >= 1 / (remaining - (max_terms-1)/max_d_limit)
        # Also, 1/d < remaining (strict, since we need at least max_terms-1 more positive terms)
        # Actually, 1/d <= remaining - (max_terms-1)/max_d_limit ensures feasibility
        thresh = remaining - (max_terms - 1) * Fraction(1, max_d_limit)
        if thresh <= 0:
            return None
        upper_from_thresh = (Fraction(1, 1) / thresh).numerator // (Fraction(1, 1) / thresh).denominator
        # Be conservative: upper = min(max_d_limit, upper_from_thresh)
        upper = min(max_d_limit, upper_from_thresh)
        
        if lower > upper:
            return None

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

    # Try to find solutions with increasing k
    # Start from k=2 and go up
    for k in range(2, 12):
        result = dfs(Fraction(1, 1), 5, k, [], 1000000)
        if result is not None:
            # Verify
            if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                if sum(Fraction(1, d) for d in result) == 1:
                    return result

    # Fallback: DP on small LCMs
    def solve_dp(target_lcm):
        candidates = []
        for d in range(5, target_lcm + 1):
            if target_lcm % d == 0:
                candidates.append(d)
        
        if not candidates:
            return None
            
        INF = float('inf')
        dp = [INF] * (target_lcm + 1)
        parent = [-1] * (target_lcm + 1)
        dp[0] = 0
        
        for d in candidates:
            w = target_lcm // d
            for s in range(target_lcm, w - 1, -1):
                if dp[s - w] + 1 < dp[s]:
                    dp[s] = dp[s - w] + 1
                    parent[s] = d
        
        if dp[target_lcm] == INF:
            return None
            
        result = []
        s = target_lcm
        while s > 0:
            d = parent[s]
            if d == -1:
                return None
            result.append(d)
            s -= target_lcm // d
        return result

    for lcm in [2520, 27720, 30030, 55440]:
        if time.time() > deadline:
            break
        result = solve_dp(lcm)
        if result is not None:
            if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                if sum(Fraction(1, d) for d in result) == 1:
                    return result

    raise RuntimeError("no representation found")
