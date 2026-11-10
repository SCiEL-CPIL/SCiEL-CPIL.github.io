# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _compute_mobius(n):
    """Compute the Möbius function mu(k) for k = 1..n."""
    mu = np.ones(n + 1, dtype=np.float64)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = False
            mu[i * i::i * i] = 0.0
            mu[i * i::i] = -mu[i * i::i]
    # mu(1) = 1
    mu[1] = 1.0
    for i in range(2, n + 1):
        if is_prime[i]:
            mu[i::i] = -mu[i::i]
    return mu


def _compute_mertens_step(mu, K):
    """
    Compute M(x) = sum_{k=1}^{min(x,K)} mu(k) * floor(x/k)
    for all x in 1..10K efficiently.
    
    We use the fact that M(x) is a step function that only changes at multiples of k.
    For each k, floor(x/k) changes at x = k, 2k, 3k, ...
    """
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.float64)
    mu = mu[1:K+1]  # mu[1], mu[2], ..., mu[K]
    
    # For each k, add mu[k] * floor(x/k) for all x
    # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
    for k in range(1, K + 1):
        if abs(mu[k-1]) < 1e-15:
            continue
        max_j = max_x // k
        for j in range(1, max_j + 1):
            start = j * k
            end = min(start + k - 1, max_x)
            M[start:end+1] += mu[k-1] * j
    return M


def _compute_mertens_fast(mu, K):
    """
    More efficient computation of M(x) for x in 1..10K.
    Uses the identity: M(x) = sum_{k=1}^{min(x,K)} mu(k) * floor(x/k)
    We can compute this using a sieve-like approach.
    
    Actually, let's use a different approach: for each x, compute M(x) directly
    but use the fact that floor(x/k) takes only O(sqrt(x)) distinct values.
    """
    max_x = 10 * K
    mu_ext = np.zeros(K + 1, dtype=np.float64)
    mu_ext[1:K+1] = mu[1:K+1]
    
    M = np.zeros(max_x + 1, dtype=np.float64)
    
    # For each x from 1 to max_x, compute M(x) = sum_{k=1}^{min(x,K)} mu(k) * floor(x/k)
    # Use the division trick: floor(x/k) = v for k in range (x/(v+1), x/v]
    for x in range(1, max_x + 1):
        total = 0.0
        k = 1
        while k <= min(x, K):
            v = x // k
            if v == 0:
                break
            next_k = min(x // v, min(x, K)) + 1
            # Sum mu(k) for k in [k, next_k-1]
            s = np.sum(mu_ext[k:min(next_k, K+1)])
            total += v * s
            k = next_k
        M[x] = total
    return M


def _check_feasibility(M, K):
    """Check if M(x) <= 1.0001 for all x in 1..10K."""
    return np.max(M[1:]) <= 1.0001


def _compute_objective(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k"""
    if len(domain) == 0:
        return 0.0
    logs = np.log(domain.astype(np.float64))
    return -np.sum(values * logs / domain.astype(np.float64))


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes, but we want safety margin
    
    # Start with a small K and try to increase it
    # The Möbius function mu(k) is the optimal answer in the limit
    # M(x) = sum_{k<=x} mu(k) floor(x/k) = 1 for x >= 1 (actually = 1 for x>=1, 0 for x<1)
    # But with finite K, we need M(x) <= 1.0001 for x up to 10K
    
    # Let's try different values of K and see which ones are feasible
    best_score = 0.0
    best_domain = None
    best_values = None
    
    # Try K = 1, 2, 3, ... up to some limit, checking feasibility
    # For small K, we can afford to check all x up to 10K
    
    # First, let's try the seed solution
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    # We need to verify this is feasible
    # M(x) = sum_{k in domain} f(k) * floor(x/k)
    # Let's compute this
    K_seed = 30
    max_x = 10 * K_seed
    M = np.zeros(max_x + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        if abs(v) < 1e-15:
            continue
        for x in range(1, max_x + 1):
            M[x] += v * (x // k)
    if np.max(M[1:]) <= 1.0001:
        best_score = _compute_objective(domain, values)
        best_domain = domain.copy()
        best_values = values.copy()
    
    # Now try to do better using the Möbius function for larger K
    # mu(k) has the property that sum_{k<=x} mu(k) floor(x/k) = 1 for all x >= 1
    # So if we use mu(k) for k = 1..K, we get M(x) = 1 for x <= K, and for x > K,
    # M(x) = sum_{k=1}^{K} mu(k) floor(x/k) which may deviate from 1
    
    # Let's try K = 10, 20, 50, 100, ... and check feasibility
    # For each K, compute mu(1)..mu(K), then M(x) for x in 1..10K
    
    # Precompute mu for a large range
    max_K_try = 500
    mu_all = _compute_mobius(max_K_try)
    
    for K in range(2, max_K_try + 1):
        if time.time() > deadline - 60:
            break
        
        # Get mu(1)..mu(K)
        mu_K = mu_all[1:K+1]
        
        # Compute M(x) for x in 1..10K
        # M(x) = sum_{k=1}^{K} mu(k) * floor(x/k)
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.float64)
        
        # Efficient computation using the fact that floor(x/k) is piecewise constant
        # For each x, we can compute M(x) in O(sqrt(x)) time
        # But for all x up to 10K, this might be slow for large K
        
        # Alternative: for each k, add mu(k) * floor(x/k) to M(x) for all x
        # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
        for k in range(1, K + 1):
            mv = mu_K[k-1]
            if abs(mv) < 1e-15:
                continue
            for j in range(1, max_x // k + 1):
                start = j * k
                end = min(start + k - 1, max_x)
                M[start:end+1] += mv * j
        
        if not _check_feasibility(M, K):
            continue
        
        # Feasible! Compute the objective
        domain_K = np.arange(1, K + 1, dtype=np.int64)
        values_K = mu_K.copy()
        score_K = _compute_objective(domain_K, values_K)
        
        if score_K > best_score:
            best_score = score_K
            best_domain = domain_K.copy()
            best_values = values_K.copy()
    
    # If we didn't find anything better than the seed, return the seed
    if best_domain is None:
        return domain, values
    
    return best_domain, best_values


# EVOLVE-BLOCK-END
