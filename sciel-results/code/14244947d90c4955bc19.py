# EVOLVE-BLOCK-START
import numpy as np
import time

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
    deadline = start_time + 600
    
    # Maximum K allowed by problem statement
    MAX_K = 200000
    
    # Compute Mobius function mu(k) for k <= MAX_K
    mu = np.ones(MAX_K + 1, dtype=np.int64)
    mu[0] = 0
    is_prime = np.ones(MAX_K + 1, dtype=bool)
    primes = []
    for i in range(2, MAX_K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, MAX_K + 1, i):
                mu[j] *= -1
                is_prime[j] = False
            for j in range(i * i, MAX_K + 1, i * i):
                mu[j] = 0
    
    # We want to find the largest K <= MAX_K such that the constraint holds.
    # Constraint: M_new(x) = M_K(x) - H_K * x <= 1.0001 for all x in [1, 10K]
    # M_K(x) = sum_{j=1}^K mu(j) floor(x/j)
    # H_K = sum_{j=1}^K mu(j)/j
    
    # Precompute prefix sums of mu(j) and mu(j)/j?
    # Actually, we need to check the constraint for specific K.
    # Let's try K = MAX_K first.
    
    def check_constraint(K):
        if K < 1:
            return False, 0.0, np.array([]), np.array([])
        
        # Compute H_K
        H_K = np.sum(mu[1:K+1] / np.arange(1, K+1, dtype=np.float64))
        
        # Compute M_K(x) for x in [1, 10K]
        # M_K(x) = sum_{j=1}^K mu(j) floor(x/j)
        # We can compute this using the O(N log N) method
        N = 10 * K
        M_vals = np.zeros(N + 1, dtype=np.float64)
        
        # For each j, add mu(j) * floor(x/j) to M_vals
        # This is equivalent to adding mu(j) to the range [m*j, (m+1)*j - 1] with weight m?
        # No, floor(x/j) = m for x in [m*j, (m+1)*j - 1].
        # So we add mu(j) * m to M_vals[m*j : (m+1)*j].
        
        # Optimization: Only iterate j where mu(j) != 0
        # But even with all j, O(N log N) is fine for N=2e6.
        
        for j in range(1, K + 1):
            if mu[j] == 0:
                continue
            c = float(mu[j])
            # floor(x/j) takes values 1, 2, ..., floor(N/j)
            # for m in 1 to floor(N/j):
            #   start = m * j
            #   end = min((m+1) * j, N + 1)
            #   M_vals[start:end] += c * m
            
            # Vectorized update is hard due to varying m.
            # We can use a difference array for the "slope" part?
            # M_K(x) is a step function.
            # Let's just use the direct loop if it's fast enough.
            # 2e6 * log(2e6) ~ 30M operations. In Python, this might be 5-10s.
            # We might check only a few K's.
            
            max_m = N // j
            # We can do this in chunks?
            # Or just loop m.
            # To speed up, we can precompute the indices?
            # Let's try a simpler approach for the check:
            # We only need to know if max(M_K(x) - H_K*x) > 1.0001.
            
            # Actually, we can compute M_K(x) for all x in [1, N] in O(N log N) using a sieve-like sum?
            # M_K(x) = sum_{j=1}^K mu(j) floor(x/j)
            # This is not a standard sieve.
            
            # Let's use the property that we can accumulate contributions.
            # For a fixed j, the contribution to M_vals is c * floor(x/j).
            # We can update M_vals in O(N/j) time.
            # Total time O(N log N).
            
            # To avoid Python loop overhead for m, we can use numpy slicing with step?
            # No, the value added is m, which changes.
            
            # Let's just do the loop. If it's too slow, we'll reduce K or optimize.
            # We can break early if time is up.
            
            # Optimization: Use a local variable for M_vals
            mv = M_vals
            for m in range(1, max_m + 1):
                start_idx = m * j
                end_idx = min((m + 1) * j, N + 1)
                mv[start_idx:end_idx] += c * m
                
        # Now compute M_new(x) = M_vals[x] - H_K * x
        x_range = np.arange(1, N + 1, dtype=np.float64)
        M_new = M_vals[1:N+1] - H_K * x_range
        max_val = np.max(M_new)
        
        if max_val > 1.0001:
            return False, max_val, np.array([]), np.array([])
        else:
            return True, max_val, mu[1:K+1], H_K

    # Try K = MAX_K
    K = MAX_K
    valid, max_val, mu_vals, H_K = check_constraint(K)
    
    if not valid:
        # Try smaller K
        # We can step down.
        # Since the constraint is likely to hold for some K, we search.
        # The metric a is increasing with K (approaching 1).
        # So we want the largest valid K.
        # We can binary search? Not monotonic.
        # Let's try a few steps.
        step = 1000
        while K > 1 and not valid:
            K -= step
            if time.time() > deadline - 60:
                break
            valid, max_val, mu_vals, H_K = check_constraint(K)
            
        if not valid:
            # Try smaller steps
            step = 100
            while K > 1 and not valid:
                K -= step
                if time.time() > deadline - 60:
                    break
                valid, max_val, mu_vals, H_K = check_constraint(K)
            
            if not valid:
                step = 10
                while K > 1 and not valid:
                    K -= step
                    if time.time() > deadline - 60:
                        break
                    valid, max_val, mu_vals, H_K = check_constraint(K)
                
                if not valid:
                    # Fallback to a known small K or just return the seed
                    # The seed is valid.
                    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
                    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
                    return domain, values
    
    # Construct domain and values
    # Domain: indices where mu(k) != 0
    # Values: mu(k)
    # But we must return the values BEFORE the f(1) adjustment?
    # The prompt says: "returning (domain, values) ... values holds the finite reals f(k)."
    # "Evaluation: values are clipped ... Then f(1) is replaced..."
    # So we return the raw mu(k) values.
    
    mask = mu_vals != 0
    domain = np.arange(1, K + 1, dtype=np.int64)[mask]
    values = mu_vals[mask].astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
