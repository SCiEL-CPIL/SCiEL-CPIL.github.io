import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

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

    # Get a baseline solution to bound the search
    baseline = get_greedy_solution()
    max_k = len(baseline) if baseline else 256
    
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

        lower_bound = max(min_den, math.ceil(rem_den / rem_num))
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
            
            result = dfs(new_num, new_den, d + 1, terms_left - 1, current_denoms + [d])
            if result is not None:
                return result
                
            if time.time() > deadline:
                return None

        return None

    # Search for minimal k
    # We know k=1,2 are impossible for d>=5
    # Start from k=3 up to max_k
    for k in range(3, max_k + 1):
        if time.time() > deadline:
            break
        result = dfs(1, 1, 5, k, [])
        if result is not None:
            return result

    # If DFS didn't find anything better than baseline, return baseline
    if baseline:
        return baseline

    raise RuntimeError("no representation found")
