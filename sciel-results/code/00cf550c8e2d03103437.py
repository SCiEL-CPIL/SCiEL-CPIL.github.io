import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Pre-compute known solutions for small k if we can find them quickly
    # Theoretical minimum for denominators > 4 is 5 terms.
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + ...
    # Let's check small k manually or via fast search first.
    
    # k=5: Max sum 1/5+1/6+1/7+1/8+1/9 = 0.2+0.166+0.142+0.125+0.111 = 0.744 < 1. Impossible.
    # k=6: Max sum +1/10 = 0.844 < 1. Impossible.
    # k=7: Max sum +1/11 = 0.935 < 1. Impossible.
    # k=8: Max sum +1/12 = 1.018 > 1. Possible?
    # Let's try k=8, 9, 10 with a very aggressive, optimized DFS.
    
    # Optimization: Use integers instead of Fractions where possible, or optimize Fraction usage.
    # The current program uses integer arithmetic which is faster than Fraction.
    # Let's stick to integer arithmetic but add better pruning.
    
    # Pruning 1: If the remaining sum (rem) is greater than the maximum possible sum of m terms
    # with denominators > prev_den, it's impossible.
    # Max sum of m terms with distinct denominators > d is approximated by sum_{i=1}^m 1/(d+i).
    # A tighter bound is actually needed.
    
    # Pruning 2: If the remaining sum (rem) is less than the minimum possible sum of m terms
    # with denominators > prev_den, it's impossible.
    # Min sum is 0 (as denominators can be arbitrarily large), but we have a cap? No, cap is 10^7.
    # However, we need exactly m terms. The smallest sum with m terms > d is roughly m/(d+m).
    # Actually, if rem is very small, we might need very large denominators.
    
    # Let's refine the DFS from the current program.
    
    def dfs(depth, prev_den, rem_num, rem_den):
        if time.time() > deadline:
            return None
        
        if depth == k:
            if rem_num == 0:
                return []
            return None
        
        m = k - depth
        
        # rem is rem_num / rem_den
        # Lower bound for next denominator
        # 1/d <= rem => d >= 1/rem = rem_den/rem_num
        min_d = max(prev_den + 1, (rem_den + rem_num - 1) // rem_num)
        
        # Upper bound for next denominator
        # rem <= m/d => d <= m/rem = m * rem_num / rem_den
        max_d = (m * rem_num) // rem_den
        
        if min_d > max_d:
            return None
        
        # Additional Pruning:
        # The sum of the remaining m terms must be exactly rem.
        # The maximum possible sum of m terms with denominators > prev_den is:
        # sum_{i=1}^m 1/(prev_den + i)
        # If rem > this sum, return None.
        # This is a strong pruning for small k.
        
        # Calculate max possible sum of m terms starting from prev_den+1
        max_possible_sum_num = 0
        max_possible_sum_den = 1
        for i in range(1, m + 1):
            d_i = prev_den + i
            # Add 1/d_i to max_possible_sum
            # max_possible_sum = max_possible_sum + 1/d_i
            new_num = max_possible_sum_num * d_i + max_possible_sum_den
            new_den = max_possible_sum_den * d_i
            g = math.gcd(new_num, new_den)
            max_possible_sum_num = new_num // g
            max_possible_sum_den = new_den // g
        
        # Check if rem > max_possible_sum
        # rem_num/rem_den > max_possible_sum_num/max_possible_sum_den
        # rem_num * max_possible_sum_den > rem_den * max_possible_sum_num
        if rem_num * max_possible_sum_den > rem_den * max_possible_sum_num:
            return None
        
        # The minimum possible sum of m terms with denominators > prev_den is 0,
        # but we need to ensure we can form exactly rem with m terms.
        # A better lower bound: The sum of m terms with denominators >= d_min is at least
        # m / d_max? No.
        
        # Iterate possible denominators
        for d in range(min_d, max_d + 1):
            # new_rem = rem_num/rem_den - 1/d
            # new_num = rem_num*d - rem_den
            # new_den = rem_den*d
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            
            if new_num < 0:
                continue
                
            if new_num == 0:
                # If remaining sum is 0, we need m-1 more terms to sum to 0, which is impossible
                # unless m-1 == 0. But m > 1 here because depth < k.
                # Wait, if new_num == 0, rem becomes 0.
                # If m-1 == 0, we are done. But m = k-depth. If depth == k-1, m=1.
                # If m=1 and new_num==0, then we have 0 terms left to sum to 0. That's valid.
                if m == 1:
                    return [d]
                else:
                    # We need m-1 terms to sum to 0. Impossible with positive denominators.
                    continue
                    
            # Simplify new_rem
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            # Pruning: Check if remaining sum can be formed by m-1 terms
            # Max sum of m-1 terms with denom > d is (m-1)/(d+1)
            # If new_rem > (m-1)/(d+1), it's impossible.
            if m > 1:
                # new_num/new_den > (m-1)/(d+1)
                # new_num * (d+1) > new_den * (m-1)
                if new_num * (d + 1) > new_den * (m - 1):
                    continue
                    
            # Additional Pruning: Max sum of m-1 terms with denom > d
            # This is tighter than (m-1)/(d+1)
            # Sum_{i=1}^{m-1} 1/(d+i)
            # For small m, we can compute this.
            if m > 2:
                max_next_num = 0
                max_next_den = 1
                for i in range(1, m):
                    d_i = d + i
                    new_n = max_next_num * d_i + max_next_den
                    new_d = max_next_den * d_i
                    gg = math.gcd(new_n, new_d)
                    max_next_num = new_n // gg
                    max_next_den = new_d // gg
                if new_num * max_next_den > new_den * max_next_num:
                    continue
            
            res = dfs(depth + 1, d, new_num, new_den)
            if res is not None:
                return [d] + res
        
        return None

    # Start from k=8, as k=5,6,7 are impossible
    for k in range(8, 257):
        if time.time() > deadline:
            break
            
        result = dfs(0, 4, 1, 1)
        if result is not None:
            # Verify the solution
            s = 0
            for d in result:
                s += 1/d
            if abs(s - 1) < 1e-10:
                return result
            # If verification fails due to floating point, try exact
            from fractions import Fraction
            s_frac = sum(Fraction(1, d) for d in result)
            if s_frac == 1:
                return result
            # If exact check fails, continue searching (should not happen)
            
    # Fallback
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
    raise RuntimeError("no baseline representation")
