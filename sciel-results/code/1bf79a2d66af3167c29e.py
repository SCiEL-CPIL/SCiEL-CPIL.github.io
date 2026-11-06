import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate primes up to a limit
    limit_primes = 1000
    is_prime = [True] * (limit_primes + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit_primes**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_primes + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # We iterate T over the primes
    # For a given T, we need all primes <= T
    
    for idx, T in enumerate(primes):
        if time.time() > deadline:
            break
            
        # Get primes <= T
        current_primes = primes[:idx+1]
        
        # Generate all T-smooth numbers <= N-1
        smooths = set()
        
        def generate(current_val, prime_idx):
            if current_val > N - 1:
                return
            smooths.add(current_val)
            for i in range(prime_idx, len(current_primes)):
                p = current_primes[i]
                if current_val * p > N - 1:
                    break
                generate(current_val * p, i)
        
        generate(1, 0)
        
        # Check for a pair
        # We need a + b = N, a,b in smooths
        for a in smooths:
            b = N - a
            if b in smooths:
                return [a, b]
                
    # Fallback if no small T found
    return [1, 123456]
