import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 1000000
    min_n = 10000
    max_n = N - 1  # n+1 must be <= 1000000
    
    # Step 1: Generate primes up to sqrt(N) using Sieve of Eratosthenes
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Step 2: Compute powerful part for all numbers up to N
    # pp[i] will store the powerful part of i
    pp = [1] * (N + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # For each power p^k where k >= 2 and p^k <= N:
        #   For each multiple m of p^k, multiply pp[m] by p
        # This accumulates the full p^e factor for each prime p
        # where e >= 2, because:
        #   If v_p(m) = e >= 2, then m is a multiple of p^2, p^3, ..., p^e
        #   So pp[m] gets multiplied by p exactly e times (once for each k=2..e)
        #   Wait, that's not right. Let me reconsider.
        
        # Actually, the powerful part of m for prime p is p^e if e >= 2, else 1.
        # If v_p(m) = e >= 2, we want to multiply pp[m] by p^e.
        
        # Method: iterate over k from 2 upward while p^k <= N.
        # For each k, for each multiple m of p^k, we need to account for p.
        # But this would overcount. Let me think again.
        
        # Better approach: for each prime p, for each multiple m of p^2,
        # compute v_p(m) and multiply pp[m] by p^v_p(m).
        
        # Even better: iterate k from 2 while p^k <= N.
        # For each k, iterate m in range(p^k, N+1, p^k).
        # For each such m, check if v_p(m) == k (i.e., m is divisible by p^k but not p^(k+1)).
        # If so, multiply pp[m] by p^k.
        # But this requires checking divisibility by p^(k+1), which is expensive.
        
        # Alternative: for each m that is a multiple of p^2, directly compute v_p(m).
        # This is what the current code does, and it's correct.
        
        # Let's use the prime power sieve approach more carefully:
        # For k = 2, 3, 4, ... while p^k <= N:
        #   For m in range(p^k, N+1, p^k):
        #     If m is NOT a multiple of p^(k+1), then v_p(m) = k, so multiply pp[m] by p^k.
        #     But if m IS a multiple of p^(k+1), then v_p(m) > k, and we'll handle it at a higher k.
        #     So we only multiply when v_p(m) is exactly k.
        
        # This is complex. Let's stick with the direct method but optimize it.
        
        # Direct method: for m in range(p2, N+1, p2):
        #   Compute v_p(m) by dividing m by p repeatedly.
        #   Multiply pp[m] by p^v_p(m).
        
        # To optimize: we can precompute for each m the highest power of p dividing m
        # using a sieve-like approach.
        
        # Actually, let's try a different approach:
        # For each prime p, create an array v_p where v_p[m] = v_p(m) for all m.
        # We can do this by:
        #   For m in range(p, N+1, p): v_p[m] = 1
        #   For m in range(p2, N+1, p2): v_p[m] += 1
        #   For m in range(p^3, N+1, p^3): v_p[m] += 1
        #   etc.
        # Then pp[m] *= p^v_p[m] if v_p[m] >= 2.
        
        # But this uses extra memory. Let's just use the direct division method.
        
        for m in range(p2, N + 1, p2):
            temp = m
            power = p2
            while temp % p == 0:
                temp //= p
                power *= p
            pp[m] *= power
    
    # Step 3: Scan for max product
    best_n = min_n
    best_score = 0
    
    for n in range(min_n, max_n + 1):
        if time.time() - start_time > deadline:
            break
        score = pp[n] * pp[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
