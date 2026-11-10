# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function mu(k) for k = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    mu[1] = 1
    return mu


def _check_feasibility(domain, values, K_max=200000):
    """
    Check if the function is feasible: M(x) <= 1.0001 for all x in [1, 10*K].
    K is the maximum key in domain.
    """
    if len(domain) == 0:
        return True
    
    K = int(np.max(domain))
    if K > K_max:
        return False
    
    # M(x) = sum_k f(k) * floor(x/k)
    # We need to check for all integer x in [1, 10*K]
    # To be efficient, we can compute M(x) for all x at once using convolution-like approach
    # But since K can be up to 200000, 10*K = 2,000,000
    # m * 10*K could be large if m is large, but m is at most K
    
    # Efficient approach: For each k in domain, f(k) contributes f(k)*floor(x/k)
    # floor(x/k) changes at multiples of k
    # We can use a difference array approach
    
    max_x = 10 * K
    if max_x > 10 * K_max:
        return False  # Should not happen if K <= K_max
    
    # Initialize M array
    M = np.zeros(max_x + 1, dtype=np.float64)
    
    # For each k, add f(k) * floor(x/k) to M[x] for x in 1..max_x
    # floor(x/k) = q for x in [q*k, (q+1)*k - 1]
    # So we can use range updates
    
    for k, f_val in zip(domain, values):
        if f_val == 0:
            continue
        k_int = int(k)
        # floor(x/k) takes values 1, 2, ..., floor(max_x/k)
        # For q from 1 to floor(max_x/k):
        #   M[x] += f_val * q for x in [q*k, min((q+1)*k - 1, max_x)]
        
        # Using difference array for range addition
        # But here the value added is not constant, it's f_val * q
        # So we need two difference arrays: one for the constant part and one for the linear part
        # Actually, let's just do it directly but efficiently
        
        # Alternative: compute floor(x/k) for all x, then multiply and accumulate
        # This is O(max_x) per k, which is O(m * max_x) total
        # If m is large, this could be slow. But let's try for now.
        
        # More efficient: use the fact that floor(x/k) is piecewise constant
        # Number of distinct values is O(sqrt(max_x)) for each k? No, it's max_x/k
        # But we can do it in O(max_x/k) time per k
        
        q_max = max_x // k_int
        for q in range(1, q_max + 1):
            start = q * k_int
            end = min((q + 1) * k_int - 1, max_x)
            M[start:end + 1] += f_val * q
    
    # Check if any M[x] > 1.0001
    return np.max(M[1:]) <= 1.0001


def _compute_objective(domain, values):
    """Compute the objective a = -sum_k f(k) * log(k) / k."""
    if len(domain) == 0:
        return 0.0
    k = domain.astype(np.float64)
    f = values
    return float(-np.sum(f * np.log(k) / k))


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin: return before 630s
    
    # Start with the baseline solution
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    best_score = 1.0000  # Current score
    
    # Try to improve by using the Möbius function
    # The Möbius function mu(k) satisfies sum_{k<=x} mu(k) * floor(x/k) = 1 for all x >= 1
    # This means M(x) = 1 for all x, which is feasible (M(x) <= 1.0001)
    # The objective is -sum mu(k) * log(k) / k, which approaches 1 as K -> infinity
    
    # Compute Möbius function for increasing K values
    for K in [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]:
        if time.time() > deadline - 60:  # Leave 60s buffer
            break
        
        mu = _compute_mobius(K)
        
        # Domain: all k from 1 to K where mu(k) != 0
        mask = mu != 0
        domain = np.where(mask)[0]
        values = mu[mask].astype(np.float64)
        
        if len(domain) == 0:
            continue
        
        # Check feasibility
        if not _check_feasibility(domain, values):
            continue
        
        # Compute objective
        score = _compute_objective(domain, values)
        
        if score > best_score:
            best_score = score
            best_domain = domain.copy()
            best_values = values.copy()
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Also try the baseline
    baseline_score = _compute_objective(best_domain, best_values)
    # The baseline has a known score of 1.0000
    
    return best_domain, best_values


# EVOLVE-BLOCK-END
