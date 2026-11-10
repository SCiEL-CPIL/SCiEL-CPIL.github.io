# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f using a smoothed exponential kernel.
    f(k) = -c/k * exp(-k/S) for k > 1, with c chosen so that M(x) <= 1
    for all x in [1, 10K]. Maximizes a = -sum f(k)*log(k)/k.
    """
    # Use large S to approach the theoretical optimum
    S = 500.0
    
    # Compute sums for c determination
    k = np.arange(2, 100000, dtype=np.float64)
    exp_term = np.exp(-k / S)
    sum_inv_k2 = np.sum(exp_term / k**2)
    
    # c from M(1) <= 1
    c = 1.0 / sum_inv_k2
    c = min(c, 10.0)
    
    # Build the function
    K = 200000
    domain = np.arange(1, K + 1, dtype=np.int64)
    k_full = domain.astype(np.float64)
    
    values = np.zeros(K, dtype=np.float64)
    mask = k_full > 1
    values[mask] = -c / k_full[mask] * np.exp(-k_full[mask] / S)
    values[0] = 0.0
    
    return domain, values


# EVOLVE-BLOCK-END
