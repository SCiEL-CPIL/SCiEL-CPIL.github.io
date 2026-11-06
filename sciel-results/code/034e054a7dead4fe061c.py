import time
import math

def construct():
    S = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s kill

    # Step 1: Compute largest prime factor for all numbers up to S
    # lpf[n] = largest prime factor of n, lpf[1] = 1
    lpf = [0] * (S + 1)
    lpf[1] = 1
    for i in range(2, S + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, S + 1, i):
                lpf[j] = i
    
    if time.time() > deadline:
        return [1, S - 1]

    # Step 2: Bucket numbers by their lpf value
    # numbers_by_lpf[p] = list of numbers x in [1, S] with lpf[x] == p
    numbers_by_lpf = {}
    for x in range(1, S + 1):
        p = lpf[x]
        if p not in numbers_by_lpf:
            numbers_by_lpf[p] = []
        numbers_by_lpf[p].append(x)
    
    if time.time() > deadline:
        return [1, S - 1]
    
    # Step 3: Iterate K from 1 upward. Maintain a boolean array `is_smooth`
    # where is_smooth[x] = True if lpf[x] <= K.
    # At each K, add numbers whose lpf == K to is_smooth.
    # Then check if any x with lpf[x] == K has its complement S-x also smooth.
    
    is_smooth = [False] * (S + 1)
    
    # Get sorted list of distinct lpf values that actually occur
    distinct_lpfs = sorted(numbers_by_lpf.keys())
    
    for k in distinct_lpfs:
        if time.time() > deadline:
            return [1, S - 1]
        
        new_numbers = numbers_by_lpf[k]
        
        # Check: for each new number x, is S-x already smooth (lpf[S-x] < k)?
        for x in new_numbers:
            comp = S - x
            if comp >= 1 and comp <= S and is_smooth[comp]:
                return [x, comp]
        
        # Check: do any two new numbers sum to S?
        # i.e., x1 + x2 = S with x1, x2 both in new_numbers
        if len(new_numbers) > 1:
            new_set = set(new_numbers)
            for x in new_numbers:
                comp = S - x
                if comp in new_set:
                    return [x, comp]
        
        # Add new numbers to is_smooth
        for x in new_numbers:
            is_smooth[x] = True
    
    # Fallback
    return [1, S - 1]
