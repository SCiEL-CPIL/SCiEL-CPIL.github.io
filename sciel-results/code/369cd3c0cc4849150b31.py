import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    N = 10**7
    
    # Step 1: Compute Smallest Prime Factor (SPF) up to N
    # This allows O(log n) computation of sigma(n)
    
    # Using a bytearray or numpy array for SPF
    # SPF[i] will store the smallest prime factor of i
    # For primes, SPF[p] = p
    
    # Memory: 10^7 * 4 bytes (int32) = 40 MB. Acceptable.
    spf = np.zeros(N + 1, dtype=np.int32)
    
    # Sieve of Eratosthenes variant to fill SPF
    # Initialize spf[i] = i for all i
    # This is O(N).
    # spf = np.arange(N + 1, dtype=np.int32)
    # for i in range(2, int(np.sqrt(N)) + 1):
    #     if spf[i] == i: # i is prime
    #         for j in range(i*i, N + 1, i):
    #             if spf[j] == j:
    #                 spf[j] = i
    
    # More efficient initialization:
    spf = np.arange(N + 1, dtype=np.int32)
    limit = int(np.sqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:
            # Mark multiples
            # j starts at i*i
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [11, 6]

    # Helper function to compute sigma(n) using SPF
    def calc_sigma(n):
        if n <= 1:
            return n
        result = 1
        temp = n
        while temp > 1:
            p = spf[temp]
            # Sum of powers of p: 1 + p + ... + p^k
            # Find highest power of p dividing temp
            power_sum = 1
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
                power_sum += power
            result *= power_sum
        return result

    # Step 2: Search for pairs
    # Iterate k = |2a - 3b| from 1 upwards
    # For each k, find the largest b such that a is integer and 1 <= a,b <= N
    # Check if sigma(a) == sigma(b)
    # Keep track of the best error
    
    best_a = 11
    best_b = 6
    best_num = 1  # error = best_num / best_den
    best_den = 3  # 1/3
    
    max_k_to_check = 10000 # Check up to k=10000
    
    for k in range(1, max_k_to_check + 1):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
            
        # Case 1: 2a - 3b = k  => 2a = 3b + k
        # a = (3b + k) / 2
        # Conditions:
        # 1. 3b + k is even
        # 2. 1 <= a <= N  =>  2 <= 3b + k <= 2N  =>  (2-k)/3 <= b <= (2N-k)/3
        # 3. 1 <= b <= N
        
        # Max b for this case:
        b_max_1 = min(N, (2 * N - k) // 3)
        if b_max_1 >= 1:
            # Adjust for parity
            # We need (3 * b + k) % 2 == 0
            # (3 * b + k) % 2 = (b + k) % 2
            # So b % 2 == k % 2
            if (b_max_1 + k) % 2 != 0:
                b_max_1 -= 1
            
            if b_max_1 >= 1:
                b = b_max_1
                a = (3 * b + k) // 2
                if 1 <= a <= N:
                    s_a = calc_sigma(a)
                    s_b = calc_sigma(b)
                    if s_a == s_b:
                        # Found a valid pair
                        # Error = k / (2b)
                        num = k
                        den = 2 * b
                        # Compare num/den with best_num/best_den
                        # num * best_den < best_num * den
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b
                        # Since we are iterating k, and this is the max b for this k,
                        # this is the best pair for this k.
                        # Can we stop?
                        # If k=1, error is ~1e-7.
                        # Next k=2, error is ~2e-7.
                        # It is highly unlikely that a larger k gives a smaller error
                        # unless b is significantly larger, which is impossible as b is bounded.
                        # However, for safety, we can continue a few more k's or break if k is small.
                        # Let's break if we find a solution for k=1, as it's almost certainly optimal.
                        # Or just continue to be safe, but limit iterations.
                        if k == 1:
                            # Very likely optimal
                            # But let's check k=2 just in case?
                            # No, 1/(2b1) vs 2/(2b2). b1, b2 approx same. 1 < 2.
                            # So k=1 is better.
                            pass

        # Case 2: 3b - 2a = k  => 2a = 3b - k
        # a = (3b - k) / 2
        # Conditions:
        # 1. 3b - k is even
        # 2. 1 <= a <= N  =>  2 <= 3b - k <= 2N  =>  (2+k)/3 <= b <= (2N+k)/3
        # 3. 1 <= b <= N
        
        b_min_2 = max(1, (2 + k + 2) // 3) # ceil((2+k)/3)
        b_max_2 = min(N, (2 * N + k) // 3)
        
        if b_max_2 >= b_min_2:
            # Adjust for parity
            # (3 * b - k) % 2 == 0
            # (b - k) % 2 == 0
            # b % 2 == k % 2
            if (b_max_2 - k) % 2 != 0:
                b_max_2 -= 1
            
            if b_max_2 >= b_min_2:
                b = b_max_2
                a = (3 * b - k) // 2
                if 1 <= a <= N:
                    s_a = calc_sigma(a)
                    s_b = calc_sigma(b)
                    if s_a == s_b:
                        num = k
                        den = 2 * b
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a = a
                            best_b = b

    return [int(best_a), int(best_b)]
