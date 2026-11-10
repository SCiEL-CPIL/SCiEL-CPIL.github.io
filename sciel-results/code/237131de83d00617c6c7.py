# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    K_max = 200000
    K = K_max

    # 1. Compute Mobius function mu(k) for k <= K
    mu = np.ones(K + 1, dtype=np.int64)
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    mu[1] = 1
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    mu_f = mu.astype(np.float64)
    
    # Compute S_mu = sum mu(k)/k for k=1..K
    S_mu = np.sum(mu_f[1:K+1] / np.arange(1, K+1, dtype=np.float64))
    
    # Compute T = sum_{k=2}^K mu(k)*log(k)/k
    ks = np.arange(1, K+1, dtype=np.float64)
    T = np.sum(mu_f[2:K+1] * np.log(ks[2:K+1]) / ks[2:K+1])
    
    # Compute M0(x) = sum_{k=1}^K mu(k) * floor(x/k) for x in [1, 10K]
    # M(x) = c * [M0(x) - S_mu * x]
    # Need max of [M0(x) - S_mu * x] over x in [1, 10K]
    
    max_x = 10 * K
    M0 = np.zeros(max_x + 1, dtype=np.float64)
    
    # Vectorized computation of M0
    # For each k, mu(k) contributes to M0[x] for x in [k, 10K] in steps of k
    # M0[x] = sum_k mu(k) * floor(x/k)
    # = sum_k mu(k) * (number of multiples of k up to x)
    
    # More efficient: for each k, add mu(k) to M0[k::k]
    nonzero_ks = np.nonzero(mu_f[1:K+1])[0] + 1
    for k in nonzero_ks:
        M0[k:max_x+1:k] += mu_f[k]
    
    # Compute the constraint: M(x) = c * [M0(x) - S_mu * x]
    # Max of [M0(x) - S_mu * x] for x in [1, 10K]
    x_arr = np.arange(1, max_x + 1, dtype=np.float64)
    constraint_vals = M0[1:max_x+1] - S_mu * x_arr
    max_constraint = np.max(constraint_vals)
    
    # c_max from constraint
    if max_constraint > 0:
        c_max = 1.0001 / max_constraint
    else:
        c_max = 10.0  # No constraint binding
    
    # Also respect clipping: |f(k)| <= 10
    # f(k) = c*mu(k) for k>=2, f(1) = c*(1-S_mu)
    # |c*mu(k)| <= 10 => c <= 10 (since |mu|<=1)
    # |c*(1-S_mu)| <= 10 => c <= 10/|1-S_mu|
    c_clip = min(10.0, 10.0 / max(abs(1.0 - S_mu), 1e-10))
    c_max = min(c_max, c_clip)
    c_max = max(c_max, 0.0)
    
    # If T > 0, we want max c; if T < 0, we want min c (but c>=0)
    if T > 0:
        c_opt = c_max
    else:
        c_opt = 0.0  # But we need m>=1, so use minimum feasible
    
    # Verify: compute actual objective
    def compute_objective(c_val, K_val, mu_f_val, S_mu_val, T_val):
        """Compute a = -sum f(k)*log(k)/k after normalization."""
        if c_val <= 0:
            return 0.0
        # a = -c * T (derived above)
        return -c_val * T_val
    
    a_opt = compute_objective(c_opt, K, mu_f, S_mu, T)
    
    # Try a few different K values to see if we can do better
    best_c = c_opt
    best_K = K
    best_a = a_opt
    
    # Try smaller K values (sparsify)
    if time.time() < deadline - 120:
        for K_try in [50000, 100000, 150000, 80000, 120000]:
            if K_try >= K:
                continue
            mu_try = mu_f[:K_try+1]
            S_mu_try = np.sum(mu_try[1:K_try+1] / np.arange(1, K_try+1, dtype=np.float64))
            T_try = np.sum(mu_try[2:K_try+1] * np.log(np.arange(1, K_try+1, dtype=np.float64)[2:K_try+1]) / np.arange(1, K_try+1, dtype=np.float64)[2:K_try+1])
            
            max_x_try = 10 * K_try
            M0_try = np.zeros(max_x_try + 1, dtype=np.float64)
            nonzero_ks_try = np.nonzero(mu_try[1:K_try+1])[0] + 1
            for k in nonzero_ks_try:
                M0_try[k:max_x_try+1:k] += mu_try[k]
            
            x_arr_try = np.arange(1, max_x_try + 1, dtype=np.float64)
            constraint_vals_try = M0_try[1:max_x_try+1] - S_mu_try * x_arr_try
            max_constraint_try = np.max(constraint_vals_try)
            
            if max_constraint_try > 0:
                c_max_try = 1.0001 / max_constraint_try
            else:
                c_max_try = 10.0
            
            c_clip_try = min(10.0, 10.0 / max(abs(1.0 - S_mu_try), 1e-10))
            c_max_try = min(c_max_try, c_clip_try)
            c_max_try = max(c_max_try, 0.0)
            
            if T_try > 0:
                c_try = c_max_try
            else:
                c_try = 0.0
            
            a_try = compute_objective(c_try, K_try, mu_try, S_mu_try, T_try)
            
            if a_try > best_a:
                best_a = a_try
                best_c = c_try
                best_K = K_try
    
    # Build final function with best parameters
    if best_K == K:
        mu_final = mu_f
    else:
        mu_final = mu_f[:best_K+1]
    
    c_final = best_c
    
    # Verify feasibility with actual computation
    def check_feasibility(K_val, c_val, mu_f_val):
        """Check if f(k) = c_val*mu(k) for k<=K_val (with S-normalization) is feasible."""
        max_x = 10 * K_val
        M0 = np.zeros(max_x + 1, dtype=np.float64)
        
        nonzero_d = np.nonzero(mu_f_val[1:K_val+1])[0] + 1
        for d in nonzero_d:
            M0[d:max_x+1:d] += mu_f_val[d]
        
        S_mu_val = np.sum(mu_f_val[1:K_val+1] / np.arange(1, K_val+1, dtype=np.float64))
        x_arr = np.arange(1, max_x + 1, dtype=np.float64)
        M_vals = c_val * (M0[1:max_x+1] - S_mu_val * x_arr)
        return np.max(M_vals) <= 1.0001
    
    if not check_feasibility(best_K, c_final, mu_final):
        # Fallback: reduce c
        c_final = c_final * 0.99
        if not check_feasibility(best_K, c_final, mu_final):
            c_final = 1.0
            if not check_feasibility(best_K, c_final, mu_final):
                # Last resort
                return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    values = (c_final * mu_final[1:best_K+1]).astype(np.float64)
    domain = np.arange(1, best_K + 1, dtype=np.int64)
    
    return domain, values

# EVOLVE-BLOCK-END
