import math
import time

def construct():
    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    # Powerful part array, 1-indexed up to N+1
    # P[i] will store the powerful part of i
    P = [1] * (N + 2)
    
    # Sieve for primes up to sqrt(N) is not enough, we need primes up to N?
    # Actually, if p > sqrt(N), p^2 > N, so no multiples of p^2 in range.
    # So we only need primes up to sqrt(N).
    limit = int(math.isqrt(N))
    
    # Simple prime sieve up to limit
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, limit + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # Compute powerful parts
    for p in primes:
        q = p * p
        while q <= N:
            # For multiples of q, check if exponent is exactly k
            # m = j * q
            # v_p(m) = k iff (j*q) / q = j is not divisible by p
            # So j % p != 0
            for m in range(q, N + 1, q):
                # Check if m/q is not divisible by p
                # m/q = m // q
                if (m // q) % p != 0:
                    P[m] *= q
            q *= p
            # Time check inside inner loop? No, outer loop is small.
            if time.time() > deadline - 5:
                # If we run out of time, we might have incomplete P
                # But the loop is fast, unlikely to hit this.
                pass
        
    # Find max P[n] * P[n+1] for n in [10000, 1000000]
    # Note: n+1 can be 1000001, so P needs size N+2
    max_score = 0
    best_n = 10000
    
    for n in range(10000, N):
        score = P[n] * P[n + 1]
        if score > max_score:
            max_score = score
            best_n = n
        if time.time() > deadline:
            break
            
    return best_n
