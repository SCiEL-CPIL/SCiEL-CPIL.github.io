import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    N = 256
    
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Define the small primes that we will bundle
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # The optimal bundle for these small primes (derived above)
    # 256 (2^8)
    # 243 (3^5)
    # 245 (5^2 * 7)
    # 17 (17)
    # 253 (11 * 23)
    # 247 (13 * 19)
    bundled_numbers = [256, 243, 245, 17, 253, 247]
    
    # Verify bundled numbers are within range and coprime to each other
    # (They are by construction of disjoint prime factors)
    
    # Add all large primes (primes > 23)
    large_primes = [p for p in primes if p > 23]
    
    # Construct the final list
    result = [1] + bundled_numbers + large_primes
    
    # Ensure types are int
    result = [int(x) for x in result]
    
    return result
