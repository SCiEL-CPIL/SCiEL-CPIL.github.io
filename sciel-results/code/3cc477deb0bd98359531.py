# EVOLVE-BLOCK-START
import numpy as np
import time

def _mobius(n):
    """Compute Mobius function for 1..n using a linear sieve."""
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _check_feasibility(domain, values, K, sample_size=2000):
    """
    Check if M(x) <= 1.0001 for a sample of x in [1, 10K].
    M(x) = sum_k f(k) * floor(x/k)
    """
    x_max = 10 * K
    # Sample x values, focusing on larger x where truncation error might be larger
    # but also including some small x just in case.
    # For x <= K, M(x) should be close to 1 (exactly 1 if infinite sum).
    # The risk is for x > K.
    # We sample uniformly in [1, 10K].
    x_samples = np.linspace(1, x_max, sample_size, dtype=np.int64)
    
    # Ensure we check some specific points like K, 2K, etc.
    specific_x = np.arange(K, x_max + 1, K)
    x_samples = np.unique(np.concatenate([x_samples, specific_x]))
    
    max_M = -np.inf
    for x in x_samples:
        # Compute M(x) = sum_k f(k) * floor(x/k)
        # Vectorized: floor(x / domain)
        # Note: domain and values are aligned
        M_x = np.sum(values * (x // domain))
        if M_x > max_M:
            max_M = M_x
        if max_M > 1.0001:
            return False
            
    return True

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 550
    
    # Try to use the maximum K
    K = 200000
    
    mu = _mobius(K)
    
    # Include all k where mu(k) != 0
    mask = mu != 0
    domain = np.where(mask)[0]
    values = mu[domain].astype(np.float64)
    
    # Ensure 1 is in domain (it is, mu(1)=1)
    # domain[0] should be 1
    
    # Compute S = sum f(k)/k
    S = np.sum(values / domain.astype(np.float64))
    
    # Adjust f(1)
    # f(1) is at index 0 in domain (since domain[0]=1)
    # But let's be safe and find index of 1
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] -= S
    else:
        # If 1 is not in domain, we should add it? 
        # The problem says "f(1) = 0 before this if 1 is not in the domain".
        # But our domain includes 1.
        pass
    
    # Check feasibility
    # To save time, check a smaller sample first
    feasible = _check_feasibility(domain, values, K, sample_size=1000)
    
    if not feasible:
        # Fallback to smaller K
        K = 50000
        mu = _mobius(K)
        mask = mu != 0
        domain = np.where(mask)[0]
        values = mu[domain].astype(np.float64)
        
        S = np.sum(values / domain.astype(np.float64))
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            values[idx_1[0]] -= S
            
        feasible = _check_feasibility(domain, values, K, sample_size=1000)
        
        if not feasible:
            # Final fallback to seed
            K = 500
            mu = _mobius(K)
            mask = mu != 0
            domain = np.where(mask)[0]
            values = mu[domain].astype(np.float64)
            
            S = np.sum(values / domain.astype(np.float64))
            idx_1 = np.where(domain == 1)[0]
            if len(idx_1) > 0:
                values[idx_1[0]] -= S

    return domain, values

# EVOLVE-BLOCK-END
