import time
from bisect import bisect_right

def construct():
    S = 123457
    deadline = time.time() + 50

    # 1. Compute Largest Prime Factor (Lpf) for all numbers up to S
    lpf = [0] * (S + 1)
    lpf[1] = 1
    
    for i in range(2, S + 1):
        if lpf[i] == 0:
            for j in range(i, S + 1, i):
                lpf[j] = i

    # 2. Bucket numbers by their Lpf value
    buckets = {}
    for x in range(1, S + 1):
        p = lpf[x]
        if p not in buckets:
            buckets[p] = []
        buckets[p].append(x)

    valid_set = set()
    
    # 3. Iterate K from 1 to S
    for k in range(1, S + 1):
        if time.time() > deadline:
            return [1, S - 1]
            
        new_numbers = buckets.get(k, [])
        if not new_numbers:
            continue
            
        # Check if any new number has its complement already in valid_set
        for x in new_numbers:
            comp = S - x
            if comp in valid_set:
                return [x, comp]
                
        # Check if any two new numbers sum to S
        # Only necessary if len(new_numbers) > 1
        if len(new_numbers) > 1:
            new_set = set(new_numbers)
            for x in new_numbers:
                comp = S - x
                if comp in new_set:
                    return [x, comp]
                    
        # Add new numbers to valid_set
        for x in new_numbers:
            valid_set.add(x)
            
    return [1, S - 1]
