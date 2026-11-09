import time

def construct():
    start_time = time.time()
    DEADLINE = 80  # seconds, safety margin before 90s kill
    
    N = 123457
    
    # List of primes to check in increasing order
    # We need primes up to at least the optimal K.
    # If the optimal K is large, we might need more primes, but for N=123457, it's likely small.
    # Let's generate primes up to 500 to be safe.
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve_primes(500)
    
    # Function to generate all K-smooth numbers <= N
    # K-smooth means all prime factors are in the list of primes up to K
    def generate_smooth(primes_list, N):
        smooths_set = set()
        
        def gen(idx, val):
            if val > N:
                return
            smooths_set.add(val)
            if idx == len(primes_list):
                return
            p = primes_list[idx]
            # Option 1: Don't use p
            gen(idx + 1, val)
            # Option 2: Use p
            if val * p <= N:
                gen(idx, val * p)
                
        gen(0, 1)
        return smooths_set

    # Try increasing max prime factor K
    # We want to minimize max(P(a), P(b)).
    # This is equivalent to finding the smallest K such that there exist a,b with a+b=N and P(a)<=K, P(b)<=K.
    # This is the same as a,b being K-smooth.
    
    for i in range(len(primes)):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        current_primes = primes[:i+1]
        max_p = current_primes[-1]
        
        # Generate K-smooth numbers
        smooths = generate_smooth(current_primes, N)
        
        # Check for valid pair
        # To ensure we find a pair, we can iterate through the smooths
        # Since a+b=N, if a is in smooths, b=N-a must be in smooths
        for a in smooths:
            b = N - a
            if b in smooths:
                # Found a pair
                # Verify a,b are positive
                if a > 0 and b > 0:
                    return [a, b]
                    
    # Fallback: Exhaustive search with LPF sieve
    # This guarantees a solution if the smooth search fails (which shouldn't happen for small N)
    
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
        if best_max_pf == 1:
            break
            
    return [best_a, best_b]
