import time
import itertools

def construct():
    start_time = time.time()
    deadline = 50  # 50 seconds to be safe
    
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # Set for O(1) lookup
    divisor_set = set(divisors)
    
    # Try k=1
    if target in divisor_set:
        return [target]
        
    # Try k=2
    # d1 + d2 = target
    # d1 < d2
    for d1 in divisors:
        d2 = target - d1
        if d2 in divisor_set and d2 != d1:
            # Ensure distinct and positive
            if d2 > 0:
                return [d1, d2]
                
    # Try k=3
    # d1 + d2 + d3 = target
    # d1 < d2 < d3
    # We can iterate d1, d2 and check d3
    # To optimize, we can break early if sum of largest possible d3 < target - d1 - d2
    # But with only 60 divisors, simple iteration is fine.
    for i in range(len(divisors)):
        d1 = divisors[i]
        if d1 >= target:
            continue
        remaining1 = target - d1
        for j in range(i + 1, len(divisors)):
            d2 = divisors[j]
            if d2 >= remaining1:
                continue
            d3 = remaining1 - d2
            if d3 in divisor_set and d3 != d1 and d3 != d2:
                if d3 > 0:
                    return [d1, d2, d3]
                    
    # If k=3 not found, try k=4, etc.
    # Given the problem statement "minimize the number", and the seed found k=6,
    # it is highly likely k=3 or k=4 is the answer.
    # We can implement a generic search for k=4, 5... if needed.
    # But let's assume k=3 exists. If not, we fallback to a greedy or DP.
    
    # Fallback: Greedy (same as seed) but optimized
    # The seed uses greedy: pick largest divisor <= remainder.
    # This might not be optimal in terms of count, but it's a valid solution.
    # We should try to find a smaller k.
    
    # Let's try k=4
    # d1 + d2 + d3 + d4 = target
    # This is O(N^3) which is 60^3 = 216,000. Still fast.
    for i in range(len(divisors)):
        d1 = divisors[i]
        if d1 >= target:
            continue
        rem1 = target - d1
        for j in range(i + 1, len(divisors)):
            d2 = divisors[j]
            if d2 >= rem1:
                continue
            rem2 = rem1 - d2
            for k in range(j + 1, len(divisors)):
                d3 = divisors[k]
                if d3 >= rem2:
                    continue
                d4 = rem2 - d3
                if d4 in divisor_set and d4 != d1 and d4 != d2 and d4 != d3:
                    if d4 > 0:
                        return [d1, d2, d3, d4]
                        
    # If k=4 not found, try k=5
    # O(N^4) = 60^4 = 12,960,000. Might be tight in Python but doable within 50s.
    # However, it's unlikely we need k=5 if k=3 or 4 exists.
    
    # Fallback to seed's greedy approach if no small k found
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0
    return chosen
