import time

def construct():
    # Use a sieve to compute phi(n) for all n up to a limit
    # This is O(N log log N) which is very fast for N=1000000
    
    limit = 1000000
    phi = list(range(limit + 1))
    
    # Sieve of Eratosthenes variant for phi
    for i in range(2, limit + 1):
        if phi[i] == i: # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n such that phi(n) is divisible by 120
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
            
    return 14400 # Fallback (should not be reached)
