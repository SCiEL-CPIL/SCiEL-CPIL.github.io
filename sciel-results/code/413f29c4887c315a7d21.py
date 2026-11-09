# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    
    # Use a moderate K to ensure feasibility and speed
    K = 1000
    
    # Compute Möbius function mu(k) for k = 1..K
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].astype(np.float64)
    
    # Check feasibility
    # Compute S = sum f(k)/k
    S = np.sum(values / domain)
    
    # Effective f
    f_eff = values.copy()
    f_eff[0] = values[0] - S # f(1) is at index 0
    
    # Compute M(x) for x in [1, 10K]
    max_x = 10 * K
    M_max = -np.inf
    # Vectorized check for all x is O(K * 10K)
    # We can do it in chunks or just accept the cost for K=1000 (10^7 ops)
    # 10^7 numpy ops is fast.
    
    # M(x) = sum f_eff[k] * floor(x/k)
    # We can compute this for all x at once?
    # x is 1..max_x
    # k is 1..K
    # floor(x/k) is a matrix of shape (max_x, K)
    # This might be too much memory (10000 * 1000 = 10^7 floats = 80MB). OK.
    
    x_vals = np.arange(1, max_x + 1, dtype=np.int64)
    k_vals = domain.astype(np.int64)
    
    # Compute floor division matrix
    # Broadcasting: x_vals[:, None] / k_vals[None, :]
    floors = np.floor(x_vals[:, None] / k_vals[None, :]).astype(np.int64)
    
    # M(x) = floors @ f_eff
    M_vals = floors @ f_eff
    
    M_max = np.max(M_vals)
    
    if M_max > 1.0001:
        # Scale down to be feasible
        # If we scale f(k) for k>1 by c, and adjust f(1)?
        # Simplest: scale all f(k) by c?
        # If we scale all f by c, S scales by c, f'(1) scales by c.
        # M(x) scales by c.
        # a scales by c.
        # So we can just scale the whole function.
        c = 1.0 / M_max
        values = values * c
        # Note: This reduces the score, but ensures feasibility.
        
    return domain, values

# EVOLVE-BLOCK-END
