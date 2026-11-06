import time

def construct():
    """
    Return integer n in [1,1000000] with phi(n) divisible by 120.
    Minimize n.
    """
    deadline = time.time() + 50
    target_divisor = 120
    limit = 1000000
    
    # Linear Sieve for Euler's Totient Function
    # phi[i] stores phi(i)
    phi = [0] * (limit + 1)
    primes = []
    is_prime = [True] * (limit + 1)
    
    phi[1] = 1
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            phi[i] = i - 1
            
        for p in primes:
            if i * p > limit:
                break
            is_prime[i * p] = False
            if i % p == 0:
                phi[i * p] = phi[i] * p
                break
            else:
                phi[i * p] = phi[i] * (p - 1)
                
        # Check if we found the answer early
        # Since we iterate i from 2 upwards, the first i where phi[i] % 120 == 0 is the answer
        if phi[i] % target_divisor == 0:
            return i
            
    # Fallback if not found (should not happen)
    return 143
