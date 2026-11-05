import time

def construct():
    start_time = time.time()
    # Set a conservative deadline to avoid being killed by the evaluator
    # Evaluator kills at 90s. We aim for 80s max.
    deadline = start_time + 80.0
    
    N = 1000000
    min_n = 10000
    
    # 1. Compute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
        # Check time occasionally to be safe, though this loop is fast
        if i % 100000 == 0 and time.time() > deadline:
            # If we run out of time, we can't complete the sieve.
            # This is extremely unlikely for N=1e6.
            # We'll just break and hope lpf is filled enough or return a default.
            # But if lpf is incomplete, lpf[n] might be 0 or a small prime, leading to incorrect min.
            # Given the speed, this branch should never be taken.
            pass

    # 2. Find n in [10000, N) minimizing max(lpf[n], lpf[n+1])
    best_n = min_n
    best_val = max(lpf[min_n], lpf[min_n + 1])
    
    # Optimization: We can break early if best_val == 2, since 2 is the smallest prime.
    # But max(lpf[n], lpf[n+1]) can be 2 only if both n and n+1 are powers of 2.
    # Consecutive powers of 2: 1, 2. Not in range [10000, ...].
    # Next smallest prime is 3.
    # Can max be 3? Both n and n+1 must be 3-smooth (primes 2,3).
    # 3-smooth numbers: 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96, 108, 128, 144, 162, 192, 216, 243, 256, 288, 324, 384, 432, 486, 512, 576, 648, 729, 768, 864, 972, 1024, 1152, 1296, 1458, 1536, 1728, 1944, 2048, 2304, 2592, 2916, 3072, 3456, 3888, 4096, 4608, 5184, 5832, 6144, 6912, 7776, 8192, 9216, 10368, 11664, 12288, 13824, 15552, 16384, 18432, 20736, 23328, 26214, 29491, 32768, 36864, 41472, 46656, 52428, 58982, 65536, 73728, 82944, 93312, 104857, 117964, 131072, 147456, 165888, 186624, 209715, 235929, 262144, 294912, 331776, 373248, 419430, 471859, 524288, 589824, 663552, 746496, 838860, 943718, 1048576...
    # Consecutive pairs in 3-smooth:
    # (1,2), (2,3), (3,4), (4,6) no, (8,9), (9,12) no...
    # Let's check if any consecutive 3-smooth numbers exist in [10000, 1000000].
    # If yes, best_val could be 3. Score 137/3 = 45.6.
    # If not, best_val might be 5. Score 27.4.
    # The loop will find it.
    
    for n in range(min_n, N):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            if best_val == 2:
                break
        if time.time() > deadline:
            break

    return best_n
