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


def _compute_M_max(domain, values, x_max):
    """Compute max of M(x) = sum_k f(k) * floor(x/k) for x in [1, x_max]."""
    m = len(domain)
    if m == 0:
        return 0.0
    
    # M(x) = sum_k f(k) * floor(x/k)
    # Use the divisor sum approach for efficiency
    # h(j) = sum_{k|j} f(k), then M(x) = cumsum(h[1..x])
    
    if x_max > 2000000:
        # Sample approach for very large x_max
        n_samples = min(x_max, 200000)
        x_samples = np.linspace(1, x_max, n_samples, dtype=np.int64)
        M_vals = np.zeros(len(x_samples))
        for idx in range(len(x_samples)):
            x = x_samples[idx]
            M_vals[idx] = np.sum(values * (x // domain))
        return np.max(M_vals)
    else:
        h = np.zeros(x_max + 1, dtype=np.float64)
        for i in range(m):
            k = int(domain[i])
            if k > x_max:
                continue
            h[k::k] += values[i]
        M_vals = np.cumsum(h[1:])
        return np.max(M_vals)


def _compute_a(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    k_float = domain.astype(np.float64)
    return -np.sum(values * np.log(k_float) / k_float)


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    x_max = 10 * K_max
    M_max = _compute_M_max(domain, values, x_max)
    return M_max <= tolerance


def prime_number_theorem_partial_function():
    """
    Construct a partial function f that maximizes a = -sum_k f(k)*log(k)/k
    subject to M(x) = sum_k f(k)*floor(x/k) <= 1.0001 for all x in [1, 10K].
    """
    start_time = time.time()
    deadline = start_time + 580
    
    # Strategy: Set f(k) = -10 for a set S of large k's.
    # After clipping, f(1) = -sum_k f(k)/k = 10 * sum_{k in S} 1/k (since f(1) was 0).
    # Constraint: f(1) <= 10 (clipping), so 10 * sum 1/k <= 10 => sum 1/k <= 1.
    # But we also need M(x) <= 1.0001.
    # M(x) = f(1) + sum_{k in S} -10 * floor(x/k).
    # For x >= 1, floor(x/k) is 0 if k > x, else >= 1.
    # M(1) = f(1) = 10 * sum_{k in S} 1/k.
    # If any k in S is 1, floor(1/1)=1, so M(1) = 10*sum(1/k) - 10.
    # But we choose large k, so 1 not in S.
    # M(1) = 10 * sum_{k in S} 1/k.
    # We need M(1) <= 1.0001 => sum 1/k <= 0.10001.
    
    # To maximize a = -sum f(k) log(k)/k = 10 * sum_{k in S} log(k)/k.
    # We pick largest k's to maximize log(k)/k per unit of 1/k cost.
    # Actually, we want to maximize sum log(k)/k subject to sum 1/k <= 0.10001.
    # The "density" is (log(k)/k) / (1/k) = log(k).
    # So we pick k with largest log(k), i.e., largest k.
    
    K = 200000
    
    # We want to select a set S of indices from [2, K] (excluding 1)
    # such that sum_{k in S} 1/k <= 0.10001.
    # We iterate from K down to 2, adding k if it fits the budget.
    
    budget = 0.10001
    S = []
    current_sum = 0.0
    
    # Iterate from K down to 2
    for k in range(K, 1, -1):
        cost = 1.0 / k
        if current_sum + cost <= budget:
            S.append(k)
            current_sum += cost
        else:
            # Since we're going in descending order of k, and cost = 1/k is increasing
            # as k decreases, if this k doesn't fit, no smaller k will fit either
            # (because 1/j > 1/k for j < k).
            # Wait, we want to maximize sum log(k)/k.
            # The "efficiency" is log(k).
            # We are picking in order of decreasing k, which is decreasing log(k).
            # So if we can't fit k, we can't fit any j < k either (higher cost, lower efficiency).
            # So we can break.
            break
    
    if not S:
        # Fallback if we couldn't pick anything (shouldn't happen for K=200000)
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([0.0, -10.0], dtype=np.float64)
        return domain, values
    
    S = np.array(S, dtype=np.int64)
    
    # Domain is 1 plus S
    domain = np.concatenate(([1], S))
    
    # Values: f(1) = 0 (will be adjusted by evaluator), f(k) = -10 for k in S
    values = np.zeros(len(domain), dtype=np.float64)
    values[1:] = -10.0
    
    # Check feasibility
    if not _check_feasibility(domain, values, K):
        # If infeasible, try a smaller budget
        # This shouldn't happen if we computed the budget correctly,
        # but let's be safe.
        # Try reducing the set S by removing the smallest k's (lowest efficiency)
        # Actually, we just need to ensure M(x) <= 1.0001.
        # M(x) = 10*sum(1/k) - 10*sum_{k in S, k<=x} floor(x/k).
        # The max of M(x) is likely at x=1, which is 10*sum(1/k) <= 1.0001.
        # So it should be feasible.
        
        # If it's not feasible, we return the Möbius function as fallback
        mu = _compute_mobius(K)
        mask = mu[1:] != 0
        domain_fallback = np.where(mask)[0] + 1
        values_fallback = mu[domain_fallback].astype(np.float64)
        return domain_fallback, values_fallback
    
    return domain, values


# EVOLVE-BLOCK-END
