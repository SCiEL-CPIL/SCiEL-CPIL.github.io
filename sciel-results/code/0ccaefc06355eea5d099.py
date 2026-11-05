import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    def dfs(rem_num, rem_den, min_den, terms_left, current_denoms):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if rem_num == 0:
                return current_denoms[:]
            return None
        
        if terms_left == 1:
            if rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d >= min_den and d <= 10000000:
                    current_denoms.append(d)
                    return current_denoms[:]
            return None

        # Pruning: Upper bound on remaining sum
        # If rem_num/rem_den > terms_left / min_den, it's impossible.
        if rem_num * min_den > rem_den * terms_left:
            return None

        lower_bound = max(min_den, math.ceil(rem_den / rem_num))
        # Upper bound: 1/d must be at least rem_num/rem_den / terms_left
        # So d <= terms_left * rem_den / rem_num
        upper_bound = int((terms_left * rem_den) // rem_num)
        
        if lower_bound > upper_bound:
            return None

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
            
            current_denoms.append(d)
            result = dfs(new_num, new_den, d + 1, terms_left - 1, current_denoms)
            current_denoms.pop()
            
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
