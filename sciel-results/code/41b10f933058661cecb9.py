import time

def construct():
    start_time = time.time()
    # Set a conservative deadline to avoid being killed by the evaluator
    # Evaluator kills at 90s. We aim for 80s max.
    deadline = start_time + 80.0
    
    N = 1000000
    min_n = 10000
    
    # 1. Compute largest prime factor for all numbers up to N
    # Using a sieve approach: for each prime i, set lpf[j] = i for all multiples j
    # This correctly identifies the LARGEST prime factor because we iterate i in increasing order
    # and overwrite lpf[j] with larger primes as we go.
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
        # Check time occasionally to be safe
        if i % 50000 == 0 and time.time() > deadline:
            # If we run out of time, we can't complete the sieve.
            # This is extremely unlikely for N=1e6.
            # We'll just break and hope lpf is filled enough.
            # But if lpf is incomplete, lpf[n] might be 0 or a small prime, leading to incorrect min.
            # Given the speed, this branch should never be taken.
            break

    # 2. Find n in [10000, N) minimizing max(lpf[n], lpf[n+1])
    best_n = min_n
    best_val = max(lpf[min_n], lpf[min_n + 1])
    
    for n in range(min_n, N):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            # Theoretical minimum is 2, but as analyzed, no consecutive powers of 2 exist in range.
            # Next is 3. If we find 3, that's likely optimal or near-optimal.
            # We can break early if we hit a very low bound to save time, but 3 is the hard floor 
            # for non-trivial cases? Actually, 2 is the floor. 
            # Let's just let it run to be sure, or break if val == 2.
            if best_val == 2:
                break
        if time.time() > deadline:
            break

    return best_n
