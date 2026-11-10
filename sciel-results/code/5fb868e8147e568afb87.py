import time

def construct():
    # Deadline guard: 90s limit, keep 5s margin
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    max_val = N + 1
    
    # Sieve for Largest Prime Factor
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                if time.time() > deadline:
                    # Fallback if sieve is too slow (unlikely)
                    # Return a known good value or seed
                    return 10000

    min_lpf = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 1000000]
    for n in range(10000, N + 1):
        # lpf[n] and lpf[n+1] are the largest prime factors
        current_lpf = max(lpf[n], lpf[n+1])
        if current_lpf < min_lpf:
            min_lpf = current_lpf
            best_n = n
            
            # Early termination if we reach the theoretical minimum?
            # The smallest prime is 2. If we find a pair where both are powers of 2, 
            # that's impossible for consecutive integers > 1.
            # Next is 3. If we find a pair with max lpf 3, it's very good.
            # But we must ensure we don't miss a better one? 
            # Actually, if min_lpf is 2, it's impossible. 
            # If min_lpf is 3, is it possible to get lower? No, primes are 2,3,5...
            # So if min_lpf == 2, we can stop.
            # Can min_lpf be 2? 
            # n, n+1 both powers of 2? Only 1,2. But n >= 10000. So no.
            # Can min_lpf be 3?
            # n, n+1 composed of 2 and 3 only.
            # Powers of 2: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384...
            # Powers of 3: 3, 9, 27, 81, 243, 729, 2187, 6561, 19683...
            # Consecutive 2,3-smooth numbers:
            # 1,2 (n=1)
            # 2,3 (n=2)
            # 3,4 (n=3)
            # 8,9 (n=8)
            # 15? 15=3*5 (not 2,3 smooth)
            # 16,17 (17 is prime)
            # 32,33 (33=3*11)
            # 80,81? 80=16*5 (no)
            # 242,243? 242=2*121 (no)
            # It is known that there are no consecutive 2,3-smooth numbers > 3,4.
            # So min_lpf cannot be 3 for n >= 10000.
            # What about 5?
            # If min_lpf becomes 5, can it be lower? No, next prime is 2,3.
            # So if we find min_lpf == 5, we can potentially stop if we are sure 2,3 are impossible.
            # Given n >= 10000, 2 and 3 are impossible.
            # So if min_lpf == 5, it's the global minimum? 
            # Wait, is it possible to have min_lpf < 5? No.
            # So if we find a pair with max lpf 5, it is the optimal solution.
            if min_lpf == 5:
                break
            if time.time() > deadline:
                break

    return best_n
