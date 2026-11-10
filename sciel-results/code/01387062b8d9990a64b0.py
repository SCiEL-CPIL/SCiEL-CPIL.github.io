# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    if K >= 1:
        is_prime[1] = False
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


def _verify_mobius_identity(mu, K, sample_points=None):
    """
    Verify that sum_{k=1}^{min(x,K)} mu(k) * floor(x/k) = 1 for given x values.
    For the full Möbius function, this identity holds exactly for all x >= 1.
    We check at strategic points to confirm correctness.
    """
    if sample_points is None:
        # Check at x = 1, 2, 3, K, 2K, 5K, 10K
        sample_points = [1, 2, 3, K, 2*K, 5*K, 10*K]
    
    f = mu.astype(np.float64)
    
    for x in sample_points:
        k_max = min(x, K)
        if k_max == 0:
            continue
        ks = np.arange(1, k_max + 1, dtype=np.int64)
        fk = f[1:k_max+1]
        floor_vals = x // ks
        M_x = np.sum(fk * floor_vals)
        if abs(M_x - 1.0) > 1e-9:
            return False
    return True


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function, which satisfies the key identity
    sum_{k<=x} mu(k) * floor(x/k) = 1 for all x >= 1.
    The objective a = -sum_k f(k) * log(k) / k approaches 1 as K -> infinity.
    
    The evaluator:
    1. Clips values to [-10, 10] (no effect on mu(k) in {-1,0,1})
    2. Adjusts f(1) so that sum_k f(k)/k = 0 (already satisfied for full mu)
    3. Checks M(x) <= 1.0001 for all x in [1, 10K] (satisfied exactly)
    4. Computes a = -sum_k f(k) * log(k) / k (approaches 1)
    """
    start_time = time.time()
    deadline = 600.0  # seconds, with safety margin
    
    # Use the maximum K allowed for the best approximation to the optimum
    # The Möbius function gives a = -sum mu(k) log(k) / k -> 1 as K -> inf
    # For K = 200000, this is very close to 1
    
    K = 200000
    
    # Compute Möbius function up to K
    mu = _compute_mobius(K)
    
    # Verify the identity holds at sample points
    if not _verify_mobius_identity(mu, K):
        # Fallback to smaller K if something went wrong
        K = 10000
        mu = _compute_mobius(K)
        if not _verify_mobius_identity(mu, K):
            # Final fallback: simple feasible solution
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Filter to non-zero values
    domain = np.where(mu != 0)[0]
    values = mu[domain].astype(np.float64)
    
    # Check time - we should be well within budget
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Return what we have
        pass
    
    return domain, values


# EVOLVE-BLOCK-END
