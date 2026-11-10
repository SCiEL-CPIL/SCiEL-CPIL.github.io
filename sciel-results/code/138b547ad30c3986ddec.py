"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 55

    # Known optimal solution with 8 terms
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 = 1
    # Let's verify this:
    # Sum = (2520/5 + 2520/6 + 2520/7 + 2520/8 + 2520/9 + 2520/10 + 2520/11 + 2520/12) / 2520
    # = (504 + 420 + 360 + 315 + 280 + 252 + 229.09... wait, 2520/11 is not integer)
    
    # Actually, let's use a known solution:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 is NOT exactly 1.
    
    # Let's try: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12
    # LCM of 5,6,7,8,9,10,11,12 is 27720
    # 27720/5 = 5544
    # 27720/6 = 4620
    # 27720/7 = 3960
    # 27720/8 = 3465
    # 27720/9 = 3080
    # 27720/10 = 2772
    # 27720/11 = 2520
    # 27720/12 = 2310
    # Sum = 5544+4620+3960+3465+3080+2772+2520+2310 = 28271
    # 28271 / 27720 != 1
    
    # Let's use the DP approach but with a better initial guess.
    # Or just return a known valid solution if we can find one quickly.
    
    def solve_dp(target_lcm):
        # Find divisors of target_lcm that are >= 5
        candidates = []
        for d in range(5, target_lcm + 1):
            if target_lcm % d == 0:
                candidates.append(d)
        
        if not candidates:
            return None
            
        # dp[s] = min number of terms to sum to s
        # parent[s] = the denominator d that was added to reach s
        INF = float('inf')
        dp = [INF] * (target_lcm + 1)
        parent = [-1] * (target_lcm + 1)
        dp[0] = 0
        
        # 0/1 Knapsack to minimize count
        # We iterate through each candidate denominator once.
        for d in candidates:
            w = target_lcm // d
            # Iterate backwards to ensure each d is used at most once
            for s in range(target_lcm, w - 1, -1):
                if dp[s - w] + 1 < dp[s]:
                    dp[s] = dp[s - w] + 1
                    parent[s] = d
        
        if dp[target_lcm] == INF:
            return None
            
        # Reconstruct the solution
        result = []
        s = target_lcm
        while s > 0:
            d = parent[s]
            if d == -1:
                return None
            result.append(d)
            s -= target_lcm // d
            
        return result

    # Try small LCMs first
    for lcm in [2520, 27720, 360360]:
        if time.time() > deadline:
            break
        result = solve_dp(lcm)
        if result is not None:
            if len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                if sum(Fraction(1, d) for d in result) == 1:
                    return result

    # DFS Fallback
    def dfs(remaining, min_d, max_terms, current, max_d_limit):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current
        if max_terms == 0:
            return None
        if remaining <= 0:
            return None

        # Max possible sum
        max_possible = Fraction(0)
        for i in range(max_terms):
            max_possible += Fraction(1, min_d + i)
        if remaining > max_possible:
            return None

        # Min possible sum
        min_possible = max_terms * Fraction(1, max_d_limit)
        if remaining < min_possible:
            return None

        # Bounds for next denominator d
        num, den = remaining.numerator, remaining.denominator
        lower_exact = (den + num - 1) // num  # ceil(den/num)
        lower = max(min_d, lower_exact)

        # Upper bound from min constraint
        thresh = remaining - (max_terms - 1) * Fraction(1, max_d_limit)
        if thresh <= 0:
            return None
        
        upper_from_min = int(1 / float(thresh))
        upper = min(max_d_limit, upper_from_min)
        
        if upper < lower:
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

    # Search for minimum number of terms
    for k in range(8, 15):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, k, [], 10000000)
        if result is not None:
            return result

    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
