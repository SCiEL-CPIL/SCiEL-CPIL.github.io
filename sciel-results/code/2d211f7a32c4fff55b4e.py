import time

def construct():
    S = 123457
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s limit

    # 1. Precompute largest prime factor (lpf) and list of primes
    lpf = [1] * (S + 1)
    primes = []
    
    for i in range(2, S + 1):
        if lpf[i] == 1:
            primes.append(i)
            # Mark multiples - standard sieve for largest prime factor
            for j in range(i, S + 1, i):
                lpf[j] = i
                
    if time.time() > deadline:
        return [1, S - 1]

    # 2. Bucket numbers by their Lpf value
    # buckets[k] contains all x such that lpf[x] == k
    # We only care about x in [1, S//2] for the "a" side, but we need to check
    # if S-x is also smooth. Since S-x can be up to S, we need the full range.
    # However, to save memory/time, we can just store the set of numbers that are
    # "smooth" up to a certain bound K.
    
    # The inspirations suggest iterating K from 1 upwards.
    # Let's group numbers by their lpf value.
    # Note: lpf[1] = 1.
    
    numbers_by_lpf = {}
    for x in range(1, S + 1):
        p = lpf[x]
        if p not in numbers_by_lpf:
            numbers_by_lpf[p] = []
        numbers_by_lpf[p].append(x)
        
    if time.time() > deadline:
        return [1, S - 1]

    # 3. Iterate K from 1 to S
    # We maintain a set `smooth_set` of all x in [1, S] such that lpf[x] <= K.
    # At each step K, we add all x where lpf[x] == K to `smooth_set`.
    # Then we check if there exists x in `smooth_set` such that S-x is also in `smooth_set`.
    # To be efficient, we can just check the newly added numbers.
    # If x is newly added (lpf[x] == K), we check if S-x is in `smooth_set`.
    # Note: S-x might be x itself (if 2x=S, i.e., x=S/2), or S-x might have been added earlier.
    # Also, we should ensure x != S-x unless we allow a=b? The problem says "two positive integers a,b".
    # Usually a and b can be equal. 123457 is odd, so a != b is guaranteed if a+b=123457.
    
    smooth_set = set()
    distinct_lpfs = sorted(numbers_by_lpf.keys())
    
    for k in distinct_lpfs:
        if time.time() > deadline:
            # Fallback to a safe heuristic
            return [1, S - 1]
            
        new_numbers = numbers_by_lpf[k]
        if not new_numbers:
            continue
            
        # Check if any new number has its complement already in smooth_set
        for x in new_numbers:
            comp = S - x
            if comp in smooth_set:
                return [x, comp]
                
        # Check if any two new numbers sum to S
        # This is only possible if x1 != x2. Since S is odd, x1 cannot equal x2.
        # So we just need to check if for any x in new_numbers, S-x is also in new_numbers.
        # We can do this by checking if S-x is in the set of new_numbers.
        if len(new_numbers) > 1:
            new_set = set(new_numbers)
            for x in new_numbers:
                comp = S - x
                if comp in new_set:
                    return [x, comp]
                    
        # Add new numbers to smooth_set
        for x in new_numbers:
            smooth_set.add(x)
            
    # Fallback
    return [1, S - 1]
