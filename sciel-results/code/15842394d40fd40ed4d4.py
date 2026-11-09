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

def check_feasibility_fast(mu, K, max_x):
    """
    Check if M(x) <= 1.0001 for all x in [1, max_x] using difference array.
    Returns (is_feasible, M_max).
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for k in range(1, K + 1):
        mk = mu[k]
        if mk == 0:
            continue
        mk_f = float(mk)
        j_max = max_x // k
        if j_max == 0:
            continue
        j = np.arange(1, j_max + 1, dtype=np.int64)
        starts = j * k
        ends = np.minimum((j + 1) * k - 1, max_x)
        diff[starts] += mk_f * j
        diff[ends + 1] -= mk_f * j
    
    M = np.cumsum(diff)
    M = M[1:max_x + 1]
    M_max = np.max(M)
    return M_max <= 1.0001, M_max

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
    
    # Try largest K first, reduce if infeasible
    # The Möbius function satisfies sum_{k|x} mu(k) = 0 for x > 1, so M(x) = 1 for all x >= 1
    # in the infinite case. For finite K, M(x) might deviate slightly.
    # We need to check feasibility.
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # List of K values to try, from large to small
    K_list = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]
    
    for K in K_list:
        if time.time() > deadline - 5:
            break
            
        max_x = 10 * K
        
        # Check feasibility
        feasible, M_max = check_feasibility_fast(mu, K, max_x)
        
        if feasible:
            # Extract domain and values where mu(k) != 0
            mask = mu[1:K + 1] != 0
            domain = np.nonzero(mask)[0] + 1
            values = mu[1:K + 1][mask].astype(np.float64)
            
            # Compute the score a = -sum f(k) * log(k) / k
            # Note: the evaluator adjusts f(1) first, but for scoring purposes,
            # we just need to ensure the structure is correct.
            # The score is proportional to a.
            log_k = np.log(domain.astype(np.float64))
            a = -np.sum(values * log_k / domain.astype(np.float64))
            
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
                
            # If we found a feasible solution with large K, we can stop
            # since larger K should give better approximation to 1.0
            if K >= 10000:
                break
    
    # Fallback: use a small known feasible solution
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
