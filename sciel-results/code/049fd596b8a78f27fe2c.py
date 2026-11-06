import time

def construct():
    start_time = time.time()
    # Use a safety margin, e.g., 50 seconds out of 60/90
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # We need to check n and n+1, so up to N_MAX + 1
    limit = N_MAX + 1
    
    # Linear Sieve for Smallest Prime Factor (SPF) and Largest Prime Factor (LPF)
    spf = [0] * (limit + 1)
    lpf = [0] * (limit + 1)
    primes = []
    
    lpf[1] = 1
    spf[1] = 1
    
    for i in range(2, limit + 1):
        if spf[i] == 0:
            spf[i] = i
            lpf[i] = i
            primes.append(i)
        
        for p in primes:
            if p > spf[i] or i * p > limit:
                break
            
            spf[i * p] = p
            
            # Calculate LPF for i * p
            # Since p <= spf[i], p is a prime factor of i * p.
            # If i % p == 0, then p is the smallest prime factor of i (actually, p is a factor of i).
            # In the linear sieve, we break when i % p == 0.
            # This means for all p < spf[i], i % p != 0.
            # When i % p == 0, p is the smallest prime factor of i.
            # Thus, lpf[i] >= p.
            # So lpf[i * p] = lpf[i].
            if i % p == 0:
                lpf[i * p] = lpf[i]
                break
            else:
                # p is a new prime factor larger than all factors of i?
                # No, p <= spf[i]. So p is smaller than or equal to the smallest factor of i.
                # Wait, if i % p != 0, then p is not a factor of i.
                # Since p <= spf[i], p is smaller than all prime factors of i.
                # So lpf[i * p] = max(lpf[i], p) = lpf[i] because lpf[i] >= spf[i] >= p.
                # Is lpf[i] always >= spf[i]? Yes, for i > 1.
                # So lpf[i * p] = lpf[i].
                lpf[i * p] = lpf[i]
                
    # Scan for the n that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = float('inf')
    
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
            
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        
        # The largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
        # because gcd(n, n+1) = 1, so prime factors are disjoint
        current_max = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
