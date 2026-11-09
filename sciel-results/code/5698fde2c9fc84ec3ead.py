import numpy as np

def prime_number_theorem_partial_function():
    # Choose K
    K = 50000
    
    # Compute Mobius function mu(k) for k=1..K
    # mu(n) = 0 if n has a squared prime factor
    # mu(n) = (-1)^k if n is a squarefree number with k prime factors
    mu = np.ones(K + 1, dtype=np.float64)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    # mu[1] is 1 by definition, which is correct.
    # Note: The sieve above sets mu[j] for multiples of primes.
    # Standard linear sieve or Euler sieve is better, but this is O(N log log N) roughly.
    # Actually, the loop `for i in range(2, K+1)` with inner loops is not standard sieve complexity.
    # Let's use a standard efficient method or just a simple loop if K is small enough.
    # K=50000 is small.
    
    # Recompute mu properly to be sure
    mu = np.ones(K + 1, dtype=np.float64)
    for i in range(2, K + 1):
        if mu[i] == 1: # i is prime
            for j in range(i, K + 1, i):
                mu[j] *= -1
            for j in range(i*i, K + 1, i*i):
                mu[j] = 0
                
    # Compute S_mu = sum mu(k)/k
    k_vals = np.arange(1, K + 1)
    S_mu = np.sum(mu[1:] / k_vals)
    
    # Compute M(x) = sum mu(k) floor(x/k) for x in 1..10K
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.float64)
    
    # Efficient update
    # For each k, add mu[k] to M[k::k]
    # This is O(K log K)
    for k in range(1, K + 1):
        if mu[k] != 0:
            M[k:max_x+1:k] += mu[k]
            
    # Compute H(x) = M(x) - S_mu * x
    x_vals = np.arange(1, max_x + 1)
    H = M[1:] - S_mu * x_vals
    
    # Find max H(x)
    H_max = np.max(H)
    
    # Determine scale c
    if H_max > 1.0001:
        c = 1.0001 / H_max
    else:
        c = 1.0
        
    # Construct domain and values
    domain = k_vals.astype(np.int64)
    values = c * mu[1:]
    
    return domain, values
