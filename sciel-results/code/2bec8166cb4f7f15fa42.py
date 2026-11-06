import time
import math

def construct():
    deadline = time.time() + 85
    N = 1000000
    
    # We want to find n in [10000, N] minimizing max(LPF(n), LPF(n+1))
    
    # Strategy: Generate K-smooth numbers.
    # If n and n+1 are both K-smooth, their LPF is <= K.
    # We can try increasing K until we find a pair, or try a fixed large K.
    
    # Let's generate primes up to a limit K.
    # The optimal LPF is likely small.
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    def generate_smooth_numbers(N, primes):
        # Generate all numbers <= N that are composed of primes in 'primes'
        smooth = set()
        
        def helper(idx, current_val):
            if current_val > N:
                return
            if current_val >= 10000: # We only care about n >= 10000
                smooth.add(current_val)
            
            # Note: n+1 must also be in the set. 
            # We can just generate all and check later.
            
            for i in range(idx, len(primes)):
                p = primes[i]
                # Multiply by p
                if current_val * p > N:
                    break
                helper(i, current_val * p) # Same index to allow powers
                
        helper(0, 1)
        return sorted(list(smooth))

    def calc_lpf(n, primes):
        # n is known to be smooth by primes <= max(primes)
        # But we need exact LPF.
        # Trial division by primes is sufficient.
        lpf = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                lpf = p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            lpf = temp
        return lpf

    # Try different K values
    # The score 10.5 suggests the value is small.
    # Let's assume the max LPF is < 1000.
    
    best_n = None
    best_val = float('inf')
    
    # Start with a small K, increase if no pairs found or to find better ones?
    # Actually, if we find a pair with K=100, its LPF is <= 100.
    # If we use K=1000, we might find a pair with LPF 500, which is worse.
    # So we want the MINIMUM LPF.
    # So we should find ALL pairs where both numbers are smooth for some K.
    # The minimal LPF will be among these.
    
    # If we generate smooth numbers for K=1000, we include all numbers with LPF <= 1000.
    # Any pair with max LPF <= 1000 will be found.
    # If the global optimum has max LPF > 1000, we miss it.
    # But given N=1e6, it's unlikely the optimum is > 1000.
    # Let's check K=1000 first.
    
    K = 1000
    primes = get_primes(K)
    smooth_nums = generate_smooth_numbers(N, primes)
    
    # Check for consecutive pairs
    for i in range(len(smooth_nums) - 1):
        n1 = smooth_nums[i]
        n2 = smooth_nums[i+1]
        if n2 == n1 + 1:
            # Calculate exact LPF
            lpf1 = calc_lpf(n1, primes)
            lpf2 = calc_lpf(n2, primes)
            val = max(lpf1, lpf2)
            if val < best_val:
                best_val = val
                best_n = n1
            if time.time() > deadline:
                break
    
    # If no pair found, or if we suspect a better one exists with higher K?
    # If best_n is None, we need a fallback.
    # Also, what if the optimum has LPF > 1000?
    # Let's try K=5000 if best_n is None or if we have time.
    
    if best_n is None or time.time() < deadline - 5:
        K2 = 5000
        primes2 = get_primes(K2)
        smooth_nums2 = generate_smooth_numbers(N, primes2)
        
        for i in range(len(smooth_nums2) - 1):
            n1 = smooth_nums2[i]
            n2 = smooth_nums2[i+1]
            if n2 == n1 + 1:
                lpf1 = calc_lpf(n1, primes2)
                lpf2 = calc_lpf(n2, primes2)
                val = max(lpf1, lpf2)
                if val < best_val:
                    best_val = val
                    best_n = n1
            if time.time() > deadline:
                break

    # Fallback: if still None, do a random search or the original method
    if best_n is None:
        # Use the original method as fallback
        # But it's slow. Let's just return a reasonable guess or do a quick scan.
        # A quick scan of random points?
        # Or just return 10000?
        # Better: run the original SPF logic but optimized?
        # The original logic is < 1s.
        
        # Let's implement a fast SPF for fallback
        spf = list(range(N + 2))
        for i in range(2, int(math.isqrt(N)) + 1):
            if spf[i] == i:
                for j in range(i * i, N + 2, i):
                    if spf[j] == j:
                        spf[j] = i
        lpf = [0] * (N + 2)
        lpf[1] = 1
        for i in range(2, N + 2):
            p = spf[i]
            lpf[i] = max(p, lpf[i // p])
            
        for n in range(10000, N + 1):
            val = lpf[n] if lpf[n] > lpf[n+1] else lpf[n+1]
            if val < best_val:
                best_val = val
                best_n = n
            if time.time() > deadline:
                break

    return best_n
