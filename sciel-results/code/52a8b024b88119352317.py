import numpy as np
import math
import time

def construct():
    start = time.time()
    deadline = start + 50  # Safety margin
    
    N = 10_000_000
    
    # Phase 1: Build Smallest Prime Factor (SPF) sieve
    # This allows O(log n) factorization and sigma calculation
    spf = np.arange(N + 1, dtype=np.int32)
    # Optimize sieve: only iterate up to sqrt(N)
    limit = int(math.isqrt(N))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            # Mark multiples of i
            # We only need to update if not already marked by a smaller prime
            # np.where is fast but creates an array.
            # Using slice assignment with mask is tricky with 'if spf[j]==j'.
            # Standard loop in Python for SPF is slow.
            # Let's use a simpler approach:
            # Just compute sigma directly using a vectorized multiplicative sieve?
            # Or stick to the additive sieve but optimized?
            
            # Actually, the additive sieve `sigma[i::i] += i` failed due to Python loop overhead.
            # Can we do it in chunks? No.
            
            # Let's try the multiplicative sieve with NumPy.
            # sigma = np.ones(N+1, dtype=np.int64)
            # for p in primes:
            #    mask = (indices % p == 0)
            #    ...
            
            # Let's go back to SPF but implement it efficiently.
            # A pure Python SPF sieve for 10^7 is ~2-3 seconds.
            # We can do it.
            pass # Placeholder, will replace with actual implementation below

    # Re-implementation of SPF construction efficiently
    # Using a list might be faster for the inner loop logic if not vectorized perfectly
    # But NumPy is better for memory.
    
    # Let's try a different vectorized sigma computation.
    # sigma(n) = sum of divisors.
    # We can compute it using the formula:
    # sigma(n) = prod_{p^e || n} (p^{e+1}-1)/(p-1)
    
    # 1. Generate primes up to N
    # 2. For each prime p, update sigma for all multiples.
    
    # This is still a Python loop over primes.
    # Number of primes ~ 660k.
    # 660k iterations in Python is ~0.1-0.5s overhead.
    # Inside, we do NumPy operations.
    # For p=2, slice is 5M elements.
    # For p=10^7, slice is 1 element.
    # Total elements processed is sum(N/p) ~ N log log N ~ 10^7 * 3 = 3*10^7.
    # This is very fast in NumPy.
    
    # Let's implement this.
    
    # Step 1: Sieve of Eratosthenes to get primes
    # Use a boolean mask
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Step 2: Compute sigma
    sigma = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        if time.time() > deadline - 20:
            break # Fallback to partial sigma? No, we need exact.
            # If we break, sigma is wrong. 
            # But primes are sorted. Small primes contribute most.
            # However, sigma is multiplicative. If we miss large primes, sigma is wrong for numbers with those factors.
            # We must complete this.
            # The loop over 660k primes is fast enough.
            
        # Find all powers of p
        power = p
        sum_power = 1
        # We need to update sigma for numbers divisible by p^k but not p^(k+1)
        # Or simpler:
        # For each multiple m of p:
        #   let m = p^e * r, where gcd(r,p)=1
        #   sigma(m) = sigma(r) * (1 + p + ... + p^e)
        #   We can update sigma[m] by multiplying by (1+p+...+p^e) / (1+...+p^(e-1))?
        #   No, sigma is not updated incrementally in that simple way if we just iterate primes.
        
        # Correct approach:
        # Initialize sigma = np.ones(N+1)
        # For each prime p:
        #   for k in range(1, N+1, p):
        #      # k is divisible by p
        #      # We need to know the exact power of p in k
        #      # This is hard to vectorize without knowing the power.
        
        # Alternative:
        # Use the additive sieve but vectorized?
        # `sigma[i::i] += i`
        # The problem was the Python loop.
        # Can we use `np.add.at`? No.
        
        # Let's try the SPF approach again, but faster.
        # Using a Python list for spf might be faster than NumPy for the sieve logic if we can't vectorize the 'if spf[j]==j' part easily.
        # But 10^7 list of ints is 80MB. OK.
        
        # Let's discard the NumPy sigma attempt and use a Python list for SPF.
        # It's robust and fast enough.
        
        pass # Break out of this block to rewrite

    # --- REWRITTEN PHASE 1: SPF Sieve in Python List ---
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    if time.time() > deadline - 10:
        return [11, 6] # Fallback

    # Helper to compute sigma(n) using spf
    def calc_sigma(n):
        if n == 1:
            return 1
        res = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            res *= power_sum
        return res

    # Phase 2: Search
    best_a, best_b = 11, 6
    best_err_num = 4 # |2*11 - 3*6|
    best_err_den = 12 # 2*6
    
    # We want to minimize |k| / (2b)
    # k = 2a - 3b
    # a = (3b + k) / 2
    # Constraints: 1 <= a, b <= N
    # a integer => 3b + k even => b + k even (since 3b has same parity as b)
    
    max_k_search = 1000
    # Iterate k in order of |k|
    # For each |k|, check b from N downwards
    # Stop checking b for this k if error > best_err
    
    for abs_k in range(1, max_k_search + 1):
        if time.time() > deadline - 5:
            break
            
        for k in ([abs_k, -abs_k] if abs_k > 0 else [0]):
            if time.time() > deadline - 5:
                break
            
            # a = (3b + k) / 2
            # 1 <= (3b+k)/2 <= N
            # 2 - k <= 3b <= 2N - k
            # b_min = ceil((2-k)/3)
            # b_max = floor((2N-k)/3)
            
            min_b = max(1, math.ceil((2 - k) / 3))
            max_b = min(N, math.floor((2 * N - k) / 3))
            
            if min_b > max_b:
                continue
                
            # Parity check: b + k must be even
            if (min_b + k) % 2 != 0:
                min_b += 1
                
            # We want to maximize b to minimize error |k|/2b
            # Start from max_b, go down to min_b
            # But we can prune:
            # Current best error E_best = best_err_num / best_err_den
            # We need |k| / (2b) < E_best
            # => 2b > |k| / E_best
            # => b > |k| * best_err_den / (2 * best_err_num)
            
            min_b_pruned = math.floor(abs(k) * best_err_den / (2 * best_err_num)) + 1
            if min_b_pruned > min_b:
                min_b = min_b_pruned
                
            if min_b > max_b:
                continue
                
            # Ensure parity
            if (min_b + k) % 2 != 0:
                min_b += 1
                
            # Search from max_b down to min_b
            # Step is 2 to maintain parity
            b = max_b
            if (b + k) % 2 != 0:
                b -= 1
                
            checks = 0
            max_checks = 50000 # Limit checks per k to avoid timeout
            
            while b >= min_b and checks < max_checks:
                if time.time() > deadline - 5:
                    break
                checks += 1
                
                a = (3 * b + k) // 2
                # a is guaranteed to be in range by min_b/max_b logic
                # and integer by parity logic
                
                if a == b:
                    b -= 2
                    continue
                    
                sigma_a = calc_sigma(a)
                sigma_b = calc_sigma(b)
                
                if sigma_a == sigma_b:
                    err_num = abs(k)
                    err_den = 2 * b
                    # Compare err_num/err_den < best_err_num/best_err_den
                    if err_num * best_err_den < best_err_num * err_den:
                        best_err_num = err_num
                        best_err_den = err_den
                        best_a, best_b = a, b
                        # If we found k=1, it's likely optimal for large b
                        # But there might be a larger b for k=1?
                        # We are iterating b downwards, so first found is max b.
                        # So for this k, we found the best b.
                        break # Move to next k? 
                        # No, we should continue checking other k's.
                        # But for this k, we found the best b (largest).
                        # So we can break the inner while loop.
                
                b -= 2

    return [int(best_a), int(best_b)]
