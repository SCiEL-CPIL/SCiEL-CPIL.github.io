# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    t0 = time.time()
    deadline = t0 + 620  # 10s safety margin before 630s kill

    # Compute Möbius function up to K using sieve
    K = 200000
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    mu[1] = 1

    # domain: all k where mu[k] != 0
    nonzero_mask = mu[1:K + 1] != 0
    domain = np.where(nonzero_mask)[0] + 1  # 1-indexed
    values = mu[domain].astype(np.float64)

    # We'll use a scaling factor c, so f(k) = c * mu(k)
    # After clipping to [-10,10], if c <= 10, no clipping occurs for mu values in {-1,0,1}
    # So for c in [0, 10], clipped values = c * mu(k)
    # S = sum_k c*mu(k)/k = c * sum_k mu(k)/k
    # f'(1) = c*mu(1) - S = c - c*sum_k mu(k)/k = c*(1 - sum_k mu(k)/k)
    # M(x) with adjusted values:
    # M(x) = sum_k f'(k) * floor(x/k)
    #      = sum_{k!=1} c*mu(k)*floor(x/k) + f'(1)*floor(x/1)
    #      = c * sum_{k!=1} mu(k)*floor(x/k) + c*(1 - sum_k mu(k)/k) * x
    #      = c * [sum_k mu(k)*floor(x/k) - mu(1)*floor(x/1) + (1 - sum_k mu(k)/k)*x]
    #      = c * [sum_k mu(k)*floor(x/k) - x + (1 - sum_k mu(k)/k)*x]
    #      = c * [sum_k mu(k)*floor(x/k) - x + x - x*sum_k mu(k)/k]
    #      = c * [sum_k mu(k)*floor(x/k) - x*sum_k mu(k)/k]
    
    # For the full Möbius function (all k up to K), sum_k mu(k)*floor(x/k) = 1 for x <= K
    # But we only have k up to K, and x goes up to 10K, so this doesn't hold for x > K.
    
    # Let's just compute numerically with a binary search on c
    
    max_x = 10 * K
    
    # Precompute floor(x/k) contributions efficiently
    # M_c(x) = c * [sum_k mu(k)*floor(x/k) - x*sum_k mu(k)/k] for the unadjusted part
    # Actually, let me just compute g(n) = sum_{d|n} f(d) for the base case (c=1)
    # then M(x) = sum_{n<=x} g(n)
    
    g_base = np.zeros(max_x + 1, dtype=np.float64)
    for idx in range(len(domain)):
        d = domain[idx]
        f_d = values[idx]
        if f_d == 0:
            continue
        for m in range(d, max_x + 1, d):
            g_base[m] += f_d
    
    M_base = np.cumsum(g_base)
    
    # Now with scaling c and f(1) adjustment:
    # f_c(k) = c * mu(k) for k != 1
    # f_c(1) = c - c*S_base where S_base = sum_k mu(k)/k
    # M_c(x) = sum_k f_c(k) * floor(x/k)
    #        = c * sum_{k!=1} mu(k)*floor(x/k) + (c - c*S_base)*x
    #        = c * [sum_{k!=1} mu(k)*floor(x/k) + x - x*S_base]
    #        = c * [sum_k mu(k)*floor(x/k) - mu(1)*floor(x/1) + x - x*S_base]
    #        = c * [M_base(x) - x + x - x*S_base]
    #        = c * [M_base(x) - x*S_base]
    
    S_base = np.sum(values / domain)
    
    # M_c(x) = c * (M_base(x) - x * S_base)
    # We need M_c(x) <= 1.0001 for all x in [1, 10K]
    # So c <= 1.0001 / max_x (M_base(x) - x*S_base) where the term is positive
    
    adjusted_M = M_base - np.arange(1, max_x + 1) * S_base
    
    # Find max of adjusted_M where it's positive
    positive_mask = adjusted_M > 0
    if np.any(positive_mask):
        max_adj = np.max(adjusted_M[positive_mask])
    else:
        max_adj = 0
    
    if max_adj > 0:
        c_max = 1.0001 / max_adj
    else:
        c_max = 10.0  # no constraint from M
    
    # Also c <= 10 due to clipping (mu values are -1,0,1, so c*mu in [-c,c], clip at 10)
    c = min(c_max, 10.0)
    
    # Apply scaling
    values = values * c
    
    # Clip to [-10, 10]
    values = np.clip(values, -10, 10)
    
    # Compute S over clipped values
    S = np.sum(values / domain)
    
    # Adjust f(1)
    if domain[0] == 1:
        idx_1 = np.where(domain == 1)[0][0]
        values[idx_1] = values[idx_1] - S
    else:
        domain = np.insert(domain, 0, 1)
        values = np.insert(values, 0, -S)
    
    elapsed = time.time() - t0
    if elapsed > 610:
        pass  # Return what we have
    
    return domain, values


# EVOLVE-BLOCK-END
