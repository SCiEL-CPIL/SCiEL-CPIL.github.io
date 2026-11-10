# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute Mobius function mu(k) for k = 1..n using linear sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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
    mu[1] = 1
    return mu


def _evaluate(domain, values, K):
    """
    Efficiently evaluate the partial function.
    Returns (feasible, score).
    """
    # Clip values
    clipped = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k
    k_float = domain.astype(np.float64)
    S = np.sum(clipped / k_float)
    
    # Adjust f(1)
    if 1 in domain:
        idx1 = np.where(domain == 1)[0][0]
        adjusted = clipped.copy()
        adjusted[idx1] = adjusted[idx1] - S
        domain_adj = domain
    else:
        adjusted = np.append(clipped, -S)
        domain_adj = np.append(domain, 1)
    
    # Check feasibility: M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, 10K]
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    
    # Vectorized contribution: for each k, add v to g[k], g[2k], g[3k], ...
    for k, v in zip(domain_adj, adjusted):
        if v != 0.0:
            g[k::k] += v
    
    M = np.cumsum(g)
    max_M = np.max(M[1:max_x + 1])
    
    if max_M > 1.0001:
        return False, -np.inf
    
    # Compute objective: a = -sum_k f(k) * log(k) / k
    k_float = domain_adj.astype(np.float64)
    score = -np.sum(adjusted * np.log(k_float) / k_float)
    
    return True, score


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function which satisfies:
      - sum_{k|x} mu(k) = 1 if x=1, else 0  =>  M(x) = 1 for all x
      - -sum_{k<=K} mu(k)*log(k)/k -> 1 as K -> infinity
    
    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 600  # Hard limit
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try increasing K values, prioritizing larger K for better objective
    K_candidates = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    
    # First, try the largest K we can afford, then work down if needed
    # But we need to be careful about time
    
    for K in K_candidates:
        # Check if we have enough time left (reserve 30s for safety)
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 30
        if remaining <= 0:
            break
        
        # Estimate time needed: O(K log K) for sieve, O(K * 10K / K) = O(10K) for eval
        # For K=200000: sieve ~200ms, eval ~2s (vectorized)
        # Be conservative: if K > 100000, only try if we have > 10s left
        if K > 100000 and remaining < 15:
            continue
        if K > 50000 and remaining < 8:
            continue
        if K > 10000 and remaining < 4:
            continue
        
        mu = _compute_mobius(K)
        # Only include k where mu(k) != 0 (squarefree numbers)
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[1:][mask].astype(np.float64)
        
        eval_start = time.time()
        feasible, score = _evaluate(domain, values, K)
        eval_time = time.time() - eval_start
        
        if feasible and score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
        
        # If evaluation was fast, we can try larger K
        # If it was slow, be more conservative
    
    if best_domain is None:
        # Fallback to original
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    return best_domain, best_values


# EVOLVE-BLOCK-END
