"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 50  # 50s safety margin before 60s limit

    # Try k=8 to k=12 with optimized DFS
    for k in range(8, 13):
        if time.time() > deadline - 5:  # Leave 5s for fallback
            break
        result = _search_k(k, start, deadline - 5)
        if result is not None:
            return result

    # Fallback: DP over divisors of 2520 (minimizing count)
    return _dp_fallback()


def _search_k(k, start, deadline):
    """DFS for exactly k distinct denominators >= 5 summing to 1 using integer pruning."""
    
    def dfs(target_num, target_den, min_den, remaining, current):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current if target_num == 0 else None
        
        if target_num <= 0:
            return None
        
        # Pruning: Max possible sum of 'remaining' terms from min_den
        # Sum = 1/min_den + ... + 1/(min_den+remaining-1)
        # Check if target > max_sum
        # target_num/target_den > max_sum
        # target_num * max_sum_den > target_den * max_sum_num
        
        # Calculate max_sum as fraction
        max_num = 0
        max_den = 1
        for i in range(remaining):
            d = min_den + i
            g = math.gcd(max_den, d)
            max_num = max_num * d // g + max_den // g
            max_den = max_den * d // g
            g = math.gcd(max_num, max_den)
            max_num //= g
            max_den //= g
        
        if target_num * max_den > target_den * max_num:
            return None
        
        # Upper bound on d: 1/d <= target => d >= 1/target
        # Also 1/d + max_sum(remaining-1, d+1) >= target
        # Simple bound: d <= remaining / target
        max_d = min(10000000, int(remaining * target_den // target_num) + 2)
        if max_d < min_den:
            return None
        
        for d in range(min_den, max_d + 1):
            if time.time() > deadline:
                return None
            
            # new_target = target - 1/d
            # target_num/target_den - 1/d = (target_num*d - target_den) / (target_den*d)
            new_num = target_num * d - target_den
            if new_num < 0:
                break
            if new_num == 0:
                if remaining == 1:
                    return current + [d]
                else:
                    continue
            
            new_den = target_den * d
            
            # Check max sum for remaining-1 terms from d+1
            # We can skip the full fraction calculation for speed if we use float approximation
            # But integer is safer. Let's use a quick float check first.
            target_val = target_num / target_den
            d_val = 1.0 / d
            new_target_val = target_val - d_val
            max_rem_val = 0.0
            for i in range(remaining - 1):
                max_rem_val += 1.0 / (d + 1 + i)
            if new_target_val > max_rem_val + 1e-12:
                continue

            result = dfs(new_num, new_den, d + 1, remaining - 1, current + [d])
            if result is not None:
                return result
        return None
    
    return dfs(1, 1, 5, k, [])


def _dp_fallback():
    """Subset DP over divisors of 2520, minimizing count."""
    target = 2520
    # dp[sum] = (count, list)
    dp = {0: (0, [])}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        items = list(dp.items())
        for total, (count, values) in items:
            new_total = total + w
            if new_total <= target:
                new_count = count + 1
                if new_total not in dp or new_count < dp[new_total][0]:
                    dp[new_total] = (new_count, values + [d])
        if target in dp:
            return dp[target][1]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
