import math
import sys
import time

def construct():
    deadline = time.time() + 55.0
    
    # Increase recursion limit to handle deep skip chains
    sys.setrecursionlimit(200000)
    
    limit = 1000000
    
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    best = limit + 1
    
    def dfs(idx, current_n, current_phi):
        nonlocal best
        if current_n >= best:
            return
        if time.time() > deadline:
            return
            
        if current_phi % 120 == 0:
            best = current_n
            return
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # If the next prime is already >= best, we can't multiply by it.
        # We can only skip it. But if we skip it, we move to the next prime.
        # Since primes are sorted, all subsequent primes are also >= best.
        # So we can just check the current state and return.
        if p >= best:
            if current_phi % 120 == 0:
                best = current_n
            return
            
        # Option 1: Skip prime p
        dfs(idx + 1, current_n, current_phi)
        
        # Option 2: Include prime p with exponent k >= 1
        n_val = current_n
        phi_val = current_phi
        for k in range(1, 20):
            n_val *= p
            if n_val >= best:
                break
            if k == 1:
                phi_val *= (p - 1)
            else:
                phi_val *= p
                
            dfs(idx + 1, n_val, phi_val)

    dfs(0, 1, 1)
    
    return best
