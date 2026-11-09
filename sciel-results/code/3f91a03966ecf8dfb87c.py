# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum f(k) log(k) / k subject to:
    1. sum f(k) / k = 0
    2. M(x) = sum f(k) floor(x/k) <= 1 for all x in [1, 10K]
    3. f(k) in [-10, 10]
    
    Strategy:
    Formulate as a Linear Program (LP).
    Variables: f_1, ..., f_K.
    Objective: Maximize -sum c_k f_k where c_k = log(k)/k.
               (Note: The evaluator computes a = -sum f(k) log(k)/k. 
                Since we want to maximize a, we minimize sum c_k f_k in standard LP form, 
                or maximize -sum c_k f_k. 
                Wait, the evaluator clips values first. 
                The clipping is non-linear. However, for k >= 2, mu(k) is in {-1, 0, 1}.
                If we assume the optimal solution stays within [-1, 1] for k>=2 (which is likely 
                given the structure of M(x) constraints for small K or specific distributions), 
                we can ignore clipping for k>=2 initially. 
                For k=1, f(1) is determined by the constraint sum f(k)/k = 0, 
                so f(1) = -sum_{k=2}^K f(k)/k.
                We must ensure |f(1)| <= 10 and |f(k)| <= 10 for all k.
                
                Let's use scipy.optimize.linprog.
                Minimize: c^T x
                Subject to: A_ub x <= b_ub
                           A_eq x == b_eq
                           bounds
                """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 10.0

    # Heuristic choice of K. 
    # K=200,000 leads to a massive LP (200k variables, 2M constraints). 
    # SciPy's HiGHS solver might handle it, but it's risky in terms of time.
    # Let's try a moderate K first, e.g., K=5000 or K=10000, to ensure speed.
    # The score is proportional to 'a'. 
    # For mu(k), a is roughly constant. 
    # Let's start with K=2000 to be safe and fast, then scale up if time permits?
    # No, we have 630s. A single LP solve for K=2000 is instant.
    # Let's try K=5000.
    
    K = 5000
    
    # Check if we have enough time for a larger K?
    # LP complexity is roughly polynomial. 
    # Let's stick to K=5000 for robustness. If it finishes fast, we could retry with larger K, 
    # but the function signature implies a single call. 
    # Actually, let's try K=20000. It's still manageable for modern solvers.
    # But 10K constraints = 200,000 constraints. 20,000 variables.
    # HiGHS is quite fast.
    
    # Let's dynamically choose K based on a quick estimate or just pick a safe large number.
    # If K is too large, the LP might time out.
    # Let's use K=10000.
    
    K = 10000
    
    # Objective coefficients: c_k = log(k) / k
    # We want to MAXIMIZE a = -sum f(k) log(k)/k.
    # This is equivalent to MINIMIZING sum f(k) log(k)/k.
    # So c = log(k)/k.
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    c = np.log(domain) / domain
    
    # Constraints:
    # 1. Equality: sum f(k)/k = 0
    #    A_eq = [1/k for k in 1..K]
    #    b_eq = [0]
    
    # 2. Inequality: M(x) <= 1 for x in 1..10K
    #    M(x) = sum_{k=1}^K f(k) floor(x/k) <= 1
    #    A_ub is a matrix of size (10K) x K.
    #    A_ub[x-1, k-1] = floor(x/k)
    
    # 3. Bounds: -10 <= f(k) <= 10
    
    # Constructing A_ub explicitly for K=10000 and x_max=100,000 
    # results in a 100,000 x 10,000 matrix. 
    # That is 1 billion floats. 8 GB of memory. This might be too much or slow to construct.
    
    # Alternative: Use the structure. 
    # M(x) = sum_{k} f(k) floor(x/k).
    # Note that floor(x/k) is non-zero only for k <= x.
    # For x < K, the row is sparse.
    # For x >= K, the row is dense.
    
    # Let's reduce the problem size.
    # Do we need to check all x?
    # M(x) is a step function.
    # Maybe we can sample x? No, the constraint is for ALL x.
    
    # Let's try a smaller K to ensure the LP is solvable within time/memory.
    # K = 2000. x_max = 20,000. Matrix 20,000 x 2,000 = 40M floats = 320MB. This is fine.
    # Let's try K = 2000 first.
    
    K = 2000
    x_max = 10 * K
    
    # 1. Prepare LP Data
    c = np.log(domain) / domain
    
    # Equality constraint: sum f(k)/k = 0
    A_eq = (1.0 / domain).reshape(1, -1)
    b_eq = np.array([0.0])
    
    # Inequality constraints: M(x) <= 1
    # A_ub[i, j] = floor((i+1) / (j+1)) for i in 0..x_max-1, j in 0..K-1
    # This is a large matrix. Let's construct it efficiently.
    # We can use broadcasting or loops.
    # x_vals = np.arange(1, x_max + 1)
    # k_vals = np.arange(1, K + 1)
    # A_ub = np.floor(x_vals[:, None] / k_vals[None, :])
    
    # Memory check: 20000 * 2000 * 8 bytes = 320 MB. Acceptable.
    
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    
    # Construct A_ub
    # To save time, we can compute in chunks or just let numpy handle it.
    # 20k x 2k is small enough for vectorized division.
    A_ub = np.floor(x_vals[:, None] / k_vals[None, :]).astype(np.float64)
    b_ub = np.ones(x_max)
    
    # Bounds
    bounds = [(-10.0, 10.0)] * K
    
    # Solve LP
    from scipy.optimize import linprog
    
    # Method 'highs' is the default and fastest.
    # Options: max_iter, etc.
    
    try:
        # Use presolve to simplify
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9}
        )
        
        if not res.success:
            # Fallback to Mobius if LP fails
            return _mobius_fallback(K)
            
        f_opt = res.x
        
        # Verify constraints manually to be safe against solver tolerance issues
        # 1. Check sum f(k)/k = 0
        S = np.sum(f_opt / domain)
        if abs(S) > 1e-6:
             # Adjust f(1) slightly? 
             # The LP enforced this. If it's off, it's solver tolerance.
             # Let's project back? 
             # f(1) = -sum_{k=2} f(k)/k
             f_opt[0] = -np.sum(f_opt[1:] / domain[1:])
             
        # 2. Check M(x) <= 1
        # M(x) = A_ub @ f_opt
        M_vals = A_ub @ f_opt
        max_M = np.max(M_vals)
        
        if max_M > 1.0001:
             # Constraint violation. 
             # Scale down f?
             # If max_M > 1, we can scale f by 1/max_M.
             # But this reduces the objective linearly.
             # Is it better to return this scaled solution or fallback?
             # The objective is linear, so scaling preserves the direction.
             # Let's scale it to be feasible.
             scale = 1.0 / max_M
             f_opt = f_opt * scale
             # Re-check sum f/k = 0 after scaling? Yes, it stays 0.
             
        # 3. Clip values (LP bounds already enforce [-10, 10], but just in case)
        f_final = np.clip(f_opt, -10.0, 10.0)
        
        # Recalculate S with clipped values?
        # The evaluator clips values to [-10, 10] BEFORE calculating S and adjusting f(1).
        # So we must simulate the evaluator's steps.
        
        # Evaluator Step 1: Clip
        f_clipped = np.clip(f_opt, -10.0, 10.0)
        
        # Evaluator Step 2: Adjust f(1)
        # S = sum f(k)/k over clipped values
        S_eval = np.sum(f_clipped / domain)
        f_eval = f_clipped.copy()
        f_eval[0] -= S_eval
        
        # Evaluator Step 3: Check M(x)
        # M(x) = sum f_eval(k) floor(x/k)
        # We need to verify M(x) <= 1.0001 for the FINAL f_eval.
        # Note: f_eval[0] changed, so M(x) changes.
        # M(x) = f_eval[0] * floor(x/1) + sum_{k=2} f_eval(k) floor(x/k)
        # floor(x/1) = x.
        # So M(x) = f_eval[0] * x + sum_{k=2} f_eval(k) floor(x/k)
        
        # We can compute this efficiently.
        # Term 1: f_eval[0] * x
        # Term 2: sum_{k=2} f_eval(k) floor(x/k)
        
        # Let's compute M_final for all x in [1, x_max]
        # We already have A_ub which contains floor(x/k) for all k.
        # M_final = A_ub @ f_eval
        
        M_final = A_ub @ f_eval
        max_M_final = np.max(M_final)
        
        if max_M_final > 1.0001:
            # If the adjustment of f(1) caused infeasibility, 
            # it's because f(1) was pushed to a large value.
            # This can happen if sum_{k=2} f(k)/k is large.
            # In that case, we might need to restrict f(1) or solve a different LP.
            # For now, fallback.
            return _mobius_fallback(K)
            
        # Evaluator Step 4: Metric
        a = -np.sum(f_eval * np.log(domain) / domain)
        
        return domain, f_eval
        
    except Exception as e:
        print(f"LP Error: {e}")
        return _mobius_fallback(K)

def _mobius_fallback(K):
    """Fallback to Mobius function if LP fails or is too slow."""
    # Compute Mobius for K
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
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

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Clip
    values_clipped = np.clip(values, -10.0, 10.0)
    
    # Adjust f(1)
    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] -= S
    
    # Verify M(x)
    x_max = 10 * K
    diff = np.zeros(x_max + 2, dtype=np.float64)
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k != 0:
            diff[k:x_max+1:k] += f_k
            
    M = np.cumsum(diff)
    max_M = np.max(M[1:x_max+1])
    
    if max_M > 1.0001:
        # Fallback to small known good set
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small
        
    return domain, values

# EVOLVE-BLOCK-END
