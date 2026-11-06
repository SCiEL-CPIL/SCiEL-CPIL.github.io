import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0
    
    # Divisors of 5040
    divisors = []
    for i in range(1, int(5040**0.5) + 1):
        if 5040 % i == 0:
            divisors.append(i)
            if i != 5040 // i:
                divisors.append(5040 // i)
    
    divisors.sort(reverse=True)
    div_set = set(divisors)
    
    target = 4979
    
    # 1. Try to find a 4-term solution
    # We know d1 must be 2520
    d1 = 2520
    if d1 in div_set:
        rem1 = target - d1
        # d2 + d3 + d4 = rem1
        # d2 < d1
        for d2 in divisors:
            if d2 >= d1:
                continue
            if d2 > rem1:
                continue
            rem2 = rem1 - d2
            # d3 + d4 = rem2
            # d3 < d2
            for d3 in divisors:
                if d3 >= d2:
                    continue
                if d3 > rem2:
                    continue
                d4 = rem2 - d3
                if d4 <= 0:
                    break # Since d3 is decreasing? No, divisors sorted desc. 
                # Wait, divisors are sorted descending.
                # If d3 is large, d4 is small.
                # We need d4 < d3.
                # If d4 < d3, and d4 is a divisor, and distinct, we found it.
                if d4 < d3:
                    if d4 in div_set:
                        # Check distinctness
                        if len({d1, d2, d3, d4}) == 4:
                            return [d1, d2, d3, d4]
                # If d4 >= d3, we might have missed a solution where d3 is smaller?
                # No, we are iterating d3. 
                # We need to ensure we don't break too early.
                # Since divisors are sorted desc, d3 decreases.
                # rem2 is fixed. d4 = rem2 - d3 increases as d3 decreases.
                # We require d4 < d3.
                # This implies 2*d3 > rem2.
                # So we only need to check d3 > rem2/2.
                # Once d3 <= rem2/2, d4 >= d3, so condition fails.
                # So we can break when d3 <= rem2/2.
                if d3 * 2 <= rem2:
                    break
            
            if time.time() > deadline - 1.0:
                break
        
        if time.time() > deadline - 1.0:
            break

    # 2. Fallback to 5-term solution
    # Verify the hardcoded solution
    fallback = [2520, 1680, 720, 56, 3]
    if sum(fallback) == target and len(set(fallback)) == 5 and all(d in div_set for d in fallback):
        return fallback
    
    # 3. Generic greedy fallback (just in case)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                return chosen
        if time.time() > deadline - 0.5:
            break
    
    return chosen

