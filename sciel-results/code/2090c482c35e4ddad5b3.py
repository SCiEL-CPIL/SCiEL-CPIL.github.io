# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    def compute_mu(K):
        """Compute Möbius function up to K using linear sieve."""
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
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

    def compute_M(domain, values, K):
        """Compute M(x) for x in [1, 10K]."""
        max_x = 10 * K
        m = len(domain)
        
        # S = sum f(k)/k
        S = np.sum(values / domain)
        f1 = -S
        
        # Compute M(x) = f1*x + sum_{k in domain} f(k)*floor(x/k)
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        # Add f1*x term
        M += f1 * np.arange(max_x + 1, dtype=np.float64)
        
        # Add contributions from each k in domain
        for idx in range(m):
            k = int(domain[idx])
            fk = values[idx]
            for j in range(1, max_x // k + 1):
                lo = j * k
                hi = min((j + 1) * k - 1, max_x)
                M[lo:hi+1] += fk
        
        return M

    # Try different K values
    best_result = None
    best_score = -np.inf
    
    for K in [50000, 20000, 10000, 5000, 2000, 1000, 500]:
        if time.time() > deadline:
            break
        
        # Compute Möbius function
        mu = compute_mu(K)
        
        # Domain: k where mu[k] != 0, k >= 2
        mask = (mu != 0)
        mask[0] = False
        mask[1] = False
        domain = np.where(mask)[0]
        values = mu[domain].astype(np.float64)
        m = len(domain)
        
        if m < 1:
            continue
        
        # Compute M(x) to find the maximum
        M = compute_M(domain, values, K)
        
        # Find max M(x) for x >= 1
        max_M = np.max(M[1:])
        
        if max_M <= 0:
            # Can scale up arbitrarily (but limited by clipping to [-10, 10])
            # Actually, if max_M <= 0, we can scale up until values hit the clip
            # But this is complex, so let's just use scale = 1
            scale = 1.0
        else:
            # Scale so that max M(x) = 1
            scale = 1.0 / max_M
        
        # Check if scaling would cause values to exceed [-10, 10]
        max_abs_val = np.max(np.abs(values))
        if max_abs_val * scale > 10:
            scale = 10.0 / max_abs_val
        
        # Apply scaling
        values_scaled = values * scale
        
        # Verify constraint
        M_scaled = compute_M(domain, values_scaled, K)
        if np.max(M_scaled[1:]) > 1.0001:
            continue
        
        # Compute objective
        a = -np.sum(values_scaled * np.log(domain.astype(np.float64)) / domain)
        
        if a > best_score:
            best_score = a
            best_result = (domain.copy(), values_scaled.copy())
    
    if best_result is not None:
        return best_result
    
    return fallback()


# EVOLVE-BLOCK-END
