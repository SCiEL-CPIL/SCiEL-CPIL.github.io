import time

def construct():
    start_time = time.time()
    deadline = 80  # 80 seconds to be safe (90s limit)
    
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
    divisor_set = set(divisors)
    N = len(divisors)
    
    # k=1
    if target in divisor_set:
        return [target]
        
    # k=2
    for d1 in divisors:
        if d1 >= target:
            continue
        d2 = target - d1
        if d2 in divisor_set and d2 != d1 and d2 > 0:
            return [d1, d2]
            
    # k=3
    # d1 + d2 + d3 = target
    # Iterate d1, d2, check d3
    for i in range(N):
        d1 = divisors[i]
        if d1 >= target:
            continue
        rem1 = target - d1
        for j in range(i + 1, N):
            d2 = divisors[j]
            if d2 >= rem1:
                continue
            d3 = rem1 - d2
            if d3 in divisor_set and d3 != d1 and d3 != d2 and d3 > 0:
                return [d1, d2, d3]
                
    # Precompute pair sums for k=4 and k=5
    # pair_sums[sum_val] = list of tuples (d_a, d_b)
    # d_a, d_b are distinct divisors
    pair_sums = {}
    for i in range(N):
        for j in range(i + 1, N):
            s = divisors[i] + divisors[j]
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((divisors[i], divisors[j]))
            
    # k=4
    # d1 + d2 + d3 + d4 = target
    # Iterate pair (d1, d2), check if target - d1 - d2 is in pair_sums
    # and the other pair (d3, d4) is disjoint from (d1, d2)
    for i in range(N):
        d1 = divisors[i]
        if d1 >= target:
            continue
        for j in range(i + 1, N):
            d2 = divisors[j]
            if d1 + d2 >= target:
                continue
            rem = target - d1 - d2
            if rem in pair_sums:
                for d3, d4 in pair_sums[rem]:
                    # Check distinctness
                    if d3 != d1 and d3 != d2 and d4 != d1 and d4 != d2:
                        # Also d3 != d4 is guaranteed by pair_sums construction (i < j)
                        return [d1, d2, d3, d4]
                        
    # k=5
    # d1 + d2 + d3 + d4 + d5 = target
    # Iterate pair (d1, d2), rem = target - d1 - d2
    # Need d3 + d4 + d5 = rem
    # Iterate d3, check if rem - d3 is in pair_sums
    # and the pair (d4, d5) is disjoint from (d1, d2, d3)
    for i in range(N):
        d1 = divisors[i]
        if d1 >= target:
            continue
        for j in range(i + 1, N):
            d2 = divisors[j]
            if d1 + d2 >= target:
                continue
            rem12 = target - d1 - d2
            for k in range(N):
                d3 = divisors[k]
                if d3 == d1 or d3 == d2:
                    continue
                if d3 >= rem12:
                    continue
                rem23 = rem12 - d3
                if rem23 in pair_sums:
                    for d4, d5 in pair_sums[rem23]:
                        # Check distinctness
                        if d4 != d1 and d4 != d2 and d4 != d3 and d5 != d1 and d5 != d2 and d5 != d3:
                            return [d1, d2, d3, d4, d5]
                            
    # Fallback: Greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
