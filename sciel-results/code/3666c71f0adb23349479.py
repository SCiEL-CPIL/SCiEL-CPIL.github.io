# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    m = len(domain)
    if m == 0:
        return True
    
    # M(x) = sum_k f(k) * floor(x/k)
    # We need to check for all integer x in [1, 10*K_max]
    # But M(x) is piecewise constant, changing only at multiples of domain elements.
    # For efficiency, we can compute M(x) at critical points.
    
    # Actually, let's just compute M(x) for all x in [1, 10*K_max] using vectorization
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    
    # Use the fact that floor(x/k) changes at multiples of k
    # For large K_max, 10*K_max can be up to 2,000,000
    # m * 10*K_max operations could be expensive if m is large
    
    # Alternative: compute M(x) incrementally
    # M(x) = M(x-1) + sum_{k | x} f(k)
    
    x_max = 10 * K_max
    if x_max > 2000000:
        # Too large, use sampling
        x_range = np.arange(1, x_max + 1, max(1, x_max // 200000))
    else:
        x_range = np.arange(1, x_max + 1)
    
    # Compute M(x) for each x in x_range
    # M(x) = sum_k f(k) * floor(x/k)
    
    # For efficiency, use the divisor sum approach
    # M(x) = sum_{d|x} g(d) where g(d) = sum_{k|d} f(k)
    # Actually M(x) = sum_{k} f(k) * floor(x/k) = sum_{j=1}^{x} sum_{k|j} f(k)
    
    # Let's compute h(j) = sum_{k|j, k in domain} f(k) for j = 1..x_max
    # Then M(x) = sum_{j=1}^{x} h(j)
    
    if x_max > 1000000:
        # Use direct computation with numpy for speed
        # M(x) = sum_k f(k) * floor(x/k)
        # Compute for a subset of x values
        x_sample = x_range[::max(1, len(x_range)//50000)]
        M_vals = np.zeros(len(x_sample))
        for idx, x in enumerate(x_sample):
            M_vals[idx] = np.sum(values * (x // domain))
        return np.all(M_vals <= tolerance)
    else:
        # Compute h(j) for j = 1..x_max
        h = np.zeros(x_max + 1, dtype=np.float64)
        for i, k in enumerate(domain):
            # Add f(k) to h(j) for all j that are multiples of k
            h[k::k] += values[i]
        
        # M(x) = cumsum of h[1..x]
        M_vals = np.cumsum(h[1:])
        return np.all(M_vals <= tolerance)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function as the base, which asymptotically achieves the optimal
    objective value of 1.0.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s safety margin
    
    # Compute Möbius function up to K
    K = 200000
    mu = _compute_mobius(K)
    
    # Use all k where mu(k) != 0
    domain = np.where(mu[1:] != 0)[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Check feasibility
    if not _check_feasibility(domain, values, K):
        # If infeasible, try smaller K
        for K_try in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
            if time.time() > deadline:
                break
            mu_try = _compute_mobius(K_try)
            domain_try = np.where(mu_try[1:] != 0)[0] + 1
            values_try = mu_try[domain_try].astype(np.float64)
            if _check_feasibility(domain_try, values_try, K_try):
                domain, values, K = domain_try, values_try, K_try
                break
        else:
            # Fallback to the original small solution
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
