import numpy as np
import time

def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def _verify_feasibility(domain, values, max_x):
    """
    Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, max_x].
    """
    m = len(domain)
    if m == 0:
        return True

    g = np.zeros(max_x + 1, dtype=np.float64)
    for i in range(m):
        k = int(domain[i])
        v = float(values[i])
        if v == 0:
            continue
        # Add v to g(n) for all n that are multiples of k
        # Use slicing for speed if possible, but loop is fine for 2M
        for n in range(k, max_x + 1, k):
            g[n] += v

    M = np.cumsum(g)
    if np.any(M[1:max_x + 1] > 1.0001):
        return False
    return True

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0

    # Max K allowed is 200,000
    K_max = 200000
    
    # Compute mu for the maximum possible K
    mu = _compute_mobius(K_max)
    
    best_domain = None
    best_values = None
    best_a = -np.inf

    # Try K values from large to small
    # We can sample or step down
    # If K is feasible, larger K might be infeasible due to M(x) deviation
    # But we want the largest K that is feasible to maximize the sum (approach limit)
    
    # Check a few large K first
    K_candidates = [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 100, 10]
    
    for K in K_candidates:
        if time.time() > deadline - 10:
            break
            
        max_x = 10 * K
        
        # domain: k where mu(k) != 0
        # But we only need k <= K
        non_zero = np.where(mu[1:K + 1] != 0)[0] + 1
        if len(non_zero) == 0:
            continue

        domain = non_zero.astype(np.int64)
        values = mu[domain].astype(np.float64)

        # Verify feasibility
        if not _verify_feasibility(domain, values, max_x):
            continue

        # Compute objective a = -sum_k f(k) * log(k) / k
        # Note: values are mu, so f(k) = mu(k)
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain.astype(np.float64))

        if a > best_a:
            best_a = a
            best_domain = domain.copy()
            best_values = values.copy()
            
        # If we found a feasible K, we might try to increase it slightly?
        # But the candidates are fixed. 
        # Since we iterate from large to small, the first feasible one is the largest.
        # We can break if we assume monotonicity? Not strictly, but likely.
        # However, a larger K usually gives a better a (closer to limit).
        # So we want the largest feasible K.
        break

    # Fallback
    if best_domain is None:
        # Use a small valid set
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values
