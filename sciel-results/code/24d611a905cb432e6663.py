# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) * log(k) / k subject to
    sum_k f(k)/k = 0 (after adjustment) and M(x) <= 1 for x in [1, 10K].
    
    Uses the Mobius function scaled by an optimal constant.
    """
    start_time = time.time()
    deadline = start_time + 550  # Leave 50s margin
    
    # Choose K. Larger K gives better approximation to the infinite sum.
    # K=10000 is a good balance.
    K = 10000
    
    # Compute Mobius function up to K
    mu = np.ones(K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
                if (j // (i * i)) * i * i == j:
                    mu[j] = 0
    
    ks = np.arange(2, K + 1)
    mus = mu[2:K + 1]
    
    # Compute L = sum_{k=2}^K mu(k)/k
    L = np.sum(mus / ks)
    
    max_x = 10 * K
    
    # We need to compute A(x) = sum_{k=2}^K mu(k) (floor(x/k) - x/k) for x in [1, max_x]
    # A(x) = F(x) - x * L
    # F(x) = sum_{k=2}^K mu(k) floor(x/k)
    
    # Efficient computation of F(x) for all x in [1, max_x]
    # F(x) is a step function.
    # We can iterate k and add mu(k) * floor(x/k) to the array.
    # Direct vectorized addition for each k:
    # x_vals = np.arange(1, max_x + 1)
    # This is K * max_x operations. 10000 * 100000 = 10^9. Too slow in Python loop?
    # Vectorized NumPy operation:
    # F = np.zeros(max_x, dtype=np.float64)
    # for k, m in zip(ks, mus):
    #     F += m * (np.arange(1, max_x + 1) // k)
    # This creates a large temporary array each time? No, if we pre-allocate?
    # Actually, `np.arange(1, max_x + 1) // k` is an array of size 100k.
    # 10000 iterations * 100k ops = 10^9 ops. NumPy can do ~10^8 ops/s. So ~10-20 seconds.
    # This is acceptable.
    
    x_vals = np.arange(1, max_x + 1, dtype=np.float64)
    F = np.zeros(max_x, dtype=np.float64)
    
    # Optimization: Only iterate k where mu(k) != 0
    non_zero_idx = np.nonzero(mus)[0]
    ks_nz = ks[non_zero_idx]
    mus_nz = mus[non_zero_idx]
    
    for k, m in zip(ks_nz, mus_nz):
        F += m * (x_vals // k)
        
    A = F - x_vals * L
    
    # Find max A(x)
    max_A = np.max(A)
    
    if max_A <= 0:
        # If max_A <= 0, any c > 0 works?
        # If A(x) is always <= 0, then c * A(x) <= 0 <= 1 for any c > 0.
        # But we want to maximize c. Is there an upper bound?
        # The constraint is c * A(x) <= 1.
        # If A(x) < 0, c * A(x) becomes more negative as c increases.
        # So if A(x) is never positive, c can be arbitrarily large?
        # No, the objective a = -c * B. If B is negative, a grows with c.
        # However, f(1) adjustment might cause issues?
        # f(1) = -c * L.
        # M(x) = f(1) + sum_{k>1} f(k) floor(x/k) = -c L + c sum mu(k) floor(x/k) = c A(x).
        # If A(x) <= 0 for all x, then M(x) <= 0 <= 1.
        # So c can be large. But we are limited by numerical precision or "infeasible" checks?
        # Actually, if max_A <= 0, we can pick a large c, say 100.
        # But usually max_A > 0 for finite K.
        c = 100.0 
    else:
        c = 1.0 / max_A
    
    # Check time
    if time.time() > deadline - 10:
         # Return a safe fallback if running out of time
         # Use a small K result or seed
         return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    # Construct output
    # f(k) = c * mu(k) for k in [2, K]
    values = c * mus
    domain = ks
    
    # Ensure domain and values are 1D arrays of equal length
    # The evaluator handles f(1).
    
    return domain, values

# EVOLVE-BLOCK-END
