import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Pre-computed known solution for k=9
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13
    # Sum = 1879/2520 + 1/11 + 1/12 + 1/13
    # Let's verify a known small solution.
    # From literature, the minimal number of terms for 1 with denominators >= 5 is 9.
    # One such solution is: 5, 6, 7, 8, 9, 10, 11, 12, 13 ? No, sum is > 1.
    # 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12 = 54232/55440 < 1.
    # We need one more term.
    # 1 - 54232/55440 = 1208/55440 = 151/6930.
    # We need a denominator d such that 1/d = 151/6930? No, d must be integer.
    # So {5,6,7,8,9,10,11,12} + {d} is not a solution.
    
    # Let's use the DFS to find the optimal solution.
    # We know k=1..8 are impossible.
    # So we start searching from k=9.
    
    def dfs(rem_num, rem_den, min_den, terms_left, current_denoms):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if rem_num == 0:
                return current_denoms
            return None
        
        if terms_left == 1:
            if rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d >= min_den and d <= 10000000:
                    return current_denoms + [d]
            return None

        # Pruning: Upper bound on remaining sum
        # The maximum possible sum of 'terms_left' distinct reciprocals starting from min_den
        # is roughly terms_left / min_den.
        # If rem_num/rem_den > terms_left / min_den, it's impossible.
        if rem_num * min_den > rem_den * terms_left:
            return None

        lower_bound = max(min_den, math.ceil(rem_den / rem_num))
        # Upper bound: 1/d must be at least rem_num/rem_den / terms_left
        # So d <= terms_left * rem_den / rem_num
        upper_bound = int((terms_left * rem_den) // rem_num)
        
        if lower_bound > upper_bound:
            return None

        # Additional pruning: if the sum of the largest possible terms is still too small
        # This is hard to compute exactly without summing, so we rely on the simple bound above.
        
        for d in range(lower_bound, upper_bound + 1):
            if d > 10000000:
                break
            
            new_num = rem_num * d - rem_den
            if new_num < 0:
                continue
            
            new_den = rem_den * d
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g
            
            result = dfs(new_num, new_den, d + 1, terms_left - 1, current_denoms + [d])
            if result is not None:
                return result
                
            if time.time() > deadline:
                return None

        return None

    # Search for minimal k
    # We established k=1..8 are impossible.
    # Start from k=9.
    for k in range(9, 257):
        if time.time() > deadline:
            break
        result = dfs(1, 1, 5, k, [])
        if result is not None:
            return result

    # Fallback: Greedy algorithm to ensure we return something valid if DFS fails
    def get_greedy_solution():
        rem_num, rem_den = 1, 1
        denoms = []
        last_d = 4
        while rem_num > 0:
            if time.time() > deadline:
                return None
            d = max(last_d + 1, math.ceil(rem_den / rem_num))
            if d > 10000000:
                return None
            denoms.append(d)
            last_d = d
            new_num = rem_num * d - rem_den
            new_den = rem_den * d
            if new_num < 0:
                return None
            g = math.gcd(new_num, new_den)
            rem_num = new_num // g
            rem_den = new_den // g
            if len(denoms) > 256:
                return None
        if rem_num == 0:
            return denoms
        return None

    baseline = get_greedy_solution()
    if baseline:
        return baseline

    raise RuntimeError("no representation found")
