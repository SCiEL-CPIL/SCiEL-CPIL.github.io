# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_mobius(n):
    """Compute Möbius function mu(k) for k = 0..n using linear sieve."""
    mu = np.zeros(n + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
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

def compute_M_scaled(mu, K, max_x):
    """
    Compute M(x)/scale for all x in [1, max_x] where f(k) = scale*mu(k) with f(1) adjustment.
    M(x) = scale * sum_{k=1}^K mu(k) * (floor(x/k) - x/k)
    Returns the array M_scaled where M_scaled[x-1] = M(x)/scale.
    """
    k_arr = np.arange(1, K+1, dtype=np.float64)
    mu_f = mu[1:K+1].astype(np.float64)
    sum_mu_over_k = np.sum(mu_f / k_arr)
    
    # Compute A(x) = sum_{k=1}^K mu(k)*floor(x/k) using difference array
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for k in range(1, K + 1):
        mk = mu_f[k-1]
        if mk == 0.0:
            continue
        j_max = max_x // k
        if j_max == 0:
            continue
        j = np.arange(1, j_max + 1, dtype=np.int64)
        starts = j * k
        ends = np.minimum((j + 1) * k - 1, max_x)
        diff[starts] += mk * j
        diff[ends + 1] -= mk * j
    
    A = np.cumsum(diff)
    A = A[1:max_x + 1]  # A[x-1] = sum mu(k)*floor(x/k)
    
    x_arr = np.arange(1, max_x + 1, dtype=np.float64)
    M_scaled = A - x_arr * sum_mu_over_k
    
    return M_scaled

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Compute Möbius function up to max K
    max_K = 200000
    mu = compute_mobius(max_K)
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # List of K values to try, from large to small
    K_list = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]
    
    for K in K_list:
        if time.time() > deadline - 30:
            break
            
        max_x = 10 * K
        
        # Compute M(x)/scale
        M_scaled = compute_M_scaled(mu, K, max_x)
        M_max_scaled = np.max(M_scaled)
        
        if M_max_scaled <= 0:
            optimal_scale = 10.0
        else:
            optimal_scale = min(10.0, 1.0001 / M_max_scaled)
        
        if optimal_scale < 1.0:
            continue
        
        # Compute the actual values
        k_arr = np.arange(1, K+1, dtype=np.float64)
        mu_f = mu[1:K+1].astype(np.float64)
        f = optimal_scale * mu_f
        
        # Apply f(1) adjustment
        S = np.sum(f / k_arr)
        f[0] = f[0] - S  # f[0] corresponds to k=1
        
        # Extract domain and values where f(k) != 0
        mask = f != 0
        domain = np.nonzero(mask)[0] + 1
        values = f[mask].astype(np.float64)
        
        # Compute the score
        score = -np.sum(f * np.log(k_arr) / k_arr)
        
        if score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
        
        # If we found a good solution, we can stop
        if K >= 10000 and optimal_scale > 2.0:
            break
    
    # Fallback: use a small known feasible solution
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
