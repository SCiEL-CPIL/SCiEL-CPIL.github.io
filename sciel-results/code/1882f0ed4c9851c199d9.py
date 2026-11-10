# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n=1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
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


def _compute_M_vectorized(domain, values, max_x):
    """
    Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
    using a vectorized difference-array approach.
    """
    diff = np.zeros(max_x + 2, dtype=np.float64)
    
    for idx in range(len(domain)):
        k = int(domain[idx])
        f_k = values[idx]
        if f_k == 0.0:
            continue
        
        max_j = max_x // k
        if max_j == 0:
            continue
        
        starts = np.arange(1, max_j + 1, dtype=np.int64) * k
        ends = np.arange(2, max_j + 2, dtype=np.int64) * k
        ends = np.minimum(ends, max_x + 1)
        vals = f_k * np.arange(1, max_j + 1, dtype=np.float64)
        
        np.add.at(diff, starts, vals)
        np.add.at(diff, ends, -vals)
    
    M = np.cumsum(diff)
    return M[:max_x + 1]


def _evaluate_candidate(domain, values, max_x):
    """
    Evaluate a candidate (domain, values):
    - Clip to [-10, 10]
    - Apply S-correction so that sum f(k)/k = 0
    - Check M(x) <= 1.0001 for all x in [1, max_x]
    - Compute objective a = -sum f(k)*log(k)/k
    Returns (domain_corrected, values_corrected, a) or None if infeasible.
    """
    k_float = domain.astype(np.float64)
    
    # Clip to [-10, 10]
    values_clipped = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k on clipped values
    S = np.sum(values_clipped / k_float)
    
    # Apply S correction: f(1) = f(1) - S
    values_corrected = values_clipped.copy()
    # Find index where domain == 1
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values_corrected[idx_1[0]] = values_clipped[idx_1[0]] - S
    else:
        # If 1 is not in domain, f(1) = 0 before correction, so f(1) = -S
        # But we can't add to domain, so this case shouldn't arise for valid inputs
        pass
    
    # Compute M(x) for x in [1, max_x]
    M = _compute_M_vectorized(domain, values_corrected, max_x)
    
    # Check constraint: M(x) <= 1.0001 for all x in [1, max_x]
    if np.max(M[1:max_x + 1]) > 1.0001:
        return None
    
    # Compute objective
    a = -np.sum(values_corrected * np.log(k_float) / k_float)
    
    return (domain, values_corrected, a)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 90.0

    best_result = None
    best_a = -np.inf

    # Strategy: Use the Möbius function μ(k) which satisfies
    # M(x) = sum_{k=1}^{x} μ(k) floor(x/k) = 1 for all x
    # and -sum μ(k) log(k)/k → 1 as K → ∞
    
    # Try increasing K values
    for K in [200000, 100000, 50000, 20000, 10000, 5000, 2000, 1000]:
        if time.time() - start_time > deadline - safety_margin:
            break
        
        # Compute Möbius function
        mu = _compute_mobius(K)
        
        # Include all k where mu(k) != 0 to keep m smaller
        mask = mu[1:K + 1] != 0
        domain = np.where(mask)[0] + 1  # 1-indexed
        values = mu[domain].astype(np.float64)
        
        if len(domain) == 0:
            continue
        
        max_x = 10 * K
        
        # Try the raw Möbius function
        result = _evaluate_candidate(domain, values, max_x)
        if result is not None:
            dom, vals, a = result
            if a > best_a:
                best_a = a
                best_result = (dom, vals)
        
        # Try scaled versions: f(k) = c * mu(k)
        # For the Möbius function, M(x) with scaling c gives M_c(x) = c * 1 = c for x >= K
        # But for x < K, M_c(x) = c * sum_{k<=x} mu(k) floor(x/k) which is still c * 1 = c
        # Wait, that's only true when we sum over all k up to x. With truncated domain,
        # for x > K, M(x) = c * sum_{k=1}^{K} mu(k) floor(x/k) which is NOT exactly c.
        # So we need to be careful. Let's just try a few scale factors.
        
        for c in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
            if time.time() - start_time > deadline - safety_margin + 30:
                break
            values_scaled = c * values
            result = _evaluate_candidate(domain, values_scaled, max_x)
            if result is not None:
                dom, vals, a = result
                if a > best_a:
                    best_a = a
                    best_result = (dom, vals)
        
        if time.time() - start_time > deadline - safety_margin + 60:
            break

    # Also try the constant-f approach as a fallback/complement
    if time.time() - start_time < deadline - 60:
        for K in [200000, 100000, 50000]:
            if time.time() - start_time > deadline - 30:
                break
            
            domain = np.arange(1, K + 1, dtype=np.int64)
            H_K = np.sum(1.0 / domain.astype(np.float64))
            c_max = 1.0001 / (H_K - 1) if H_K > 1 else 0.001
            
            for c_frac in [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]:
                if time.time() - start_time > deadline - 15:
                    break
                c = c_max * c_frac
                values = np.full(K, -c, dtype=np.float64)
                result = _evaluate_candidate(domain, values, 10 * K)
                if result is not None:
                    dom, vals, a = result
                    if a > best_a:
                        best_a = a
                        best_result = (dom, vals)

    if best_result is None:
        # Fallback to a known feasible solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_result


# EVOLVE-BLOCK-END
