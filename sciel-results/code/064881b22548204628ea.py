# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for the 60s wall-clock limit

    def search(remaining_num, remaining_den, min_den, terms_left, current, deadline):
        if time.time() > deadline:
            return None
        
        if terms_left == 0:
            if remaining_num == 0:
                return list(current)
            return None
        
        if remaining_num == 0:
            return None
        
        if terms_left == 1:
            if remaining_den % remaining_num == 0:
                d = remaining_den // remaining_num
                if d >= min_den and d >= 5 and d <= 10000000:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
            return None
        
        # Pruning: max achievable with terms_left terms starting from min_den
        # Sum of 1/(min_den + i) for i in range(terms_left)
        max_sum_num, max_sum_den = 0, 1
        for i in range(terms_left):
            d = min_den + i
            new_num = max_sum_num * d + max_sum_den
            new_den = max_sum_den * d
            g = gcd(new_num, new_den)
            max_sum_num, max_sum_den = new_num // g, new_den // g
        
        if remaining_num * max_sum_den > max_sum_num * remaining_den:
            return None
        
        # Lower bound on d
        d_lower = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)
        
        # Upper bound on d:
        # remaining - 1/d <= max_achievable(terms_left-1, d+1)
        # max_achievable(terms_left-1, d+1) <= (terms_left-1) / (d+1)
        # remaining - 1/d <= (terms_left-1) / (d+1)
        # remaining <= 1/d + (terms_left-1)/(d+1)
        # remaining * d * (d+1) <= (d+1) + d * (terms_left-1)
        # This is a quadratic inequality in d.
        # Simpler bound: remaining <= terms_left / d  => d <= terms_left / remaining
        # d <= floor(terms_left * remaining_den / remaining_num)
        d_upper = (terms_left * remaining_den) // remaining_num
        
        if d_upper < d_lower:
            return None
            
        # Cap d_upper to a reasonable limit to avoid infinite loops if bound is loose
        # But the bound d <= k/remaining is tight.
        
        for d in range(d_lower, d_upper + 1):
            if time.time() > deadline:
                return None
            
            # 1/d <= remaining?
            if d * remaining_num < remaining_den:
                continue
            
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num < 0:
                continue
            
            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            if terms_left - 1 == 0:
                if new_num == 0:
                    current.append(d)
                    result = list(current)
                    current.pop()
                    return result
                continue
            
            current.append(d)
            result = search(new_num, new_den, d + 1, terms_left - 1, current, deadline)
            if result is not None:
                return result
            current.pop()
        
        return None

    def verify(result):
        if not result:
            return False
        total_num, total_den = 0, 1
        for d in result:
            total_num = total_num * d + total_den
            total_den = total_den * d
            g = gcd(total_num, total_den)
            total_num //= g
            total_den //= g
        if total_num != total_den:
            return False
        if len(set(result)) != len(result):
            return False
        if not all(5 <= d <= 10000000 for d in result):
            return False
        return True

    # Try 9 terms
    result = search(1, 1, 5, 9, [], deadline)
    if result and verify(result):
        return result

    # Try 10 terms
    if time.time() < deadline - 10:
        result = search(1, 1, 5, 10, [], deadline)
        if result and verify(result):
            return result

    # Fallback: greedy
    remaining_num, remaining_den = 1, 1
    result = []
    min_d = 5
    for _ in range(256):
        if remaining_num == 0:
            break
        if time.time() > deadline:
            break
        d = max(min_d, (remaining_den + remaining_num - 1) // remaining_num)
        while d * remaining_num < remaining_den:
            d += 1
        if d > 10000000:
            break
        result.append(d)
        remaining_num = remaining_num * d - remaining_den
        remaining_den = remaining_den * d
        g = gcd(remaining_num, remaining_den)
        if g > 1:
            remaining_num //= g
            remaining_den //= g
        min_d = d + 1
    
    if remaining_num == 0 and verify(result):
        return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
