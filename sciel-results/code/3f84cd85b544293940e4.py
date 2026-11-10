# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    
    # Deadline guard: 600s limit, keep 30s margin
    deadline = start_time + 570
    
    # Strategy:
    # 1. Formulate the problem as a Linear Program (LP).
    #    Variables: g(k) for k in domain D (subset of [2, K]).
    #    Objective: Maximize a = -sum_{k in D} g(k) log(k) / k.
    #    Constraints:
    #       - M(x) = sum_{k in D} g(k) (floor(x/k) - x/k) <= 1 for all x in [1, 10K].
    #         (This accounts for f(1) = -sum g(k)/k).
    #       - -10 <= g(k) <= 10.
    # 2. Solve the LP using scipy.optimize.linprog with the 'highs' method.
    # 3. Verify feasibility. If infeasible or timeout, fall back to a known valid solution.
    
    # We choose K=200 to balance computational cost and approximation quality.
    # K=200 gives 10K=2000 constraints.
    # The number of variables is the count of square-free numbers in [2, 200], which is ~120.
    # This LP size is very small for HiGHS.
    
    K = 200
    
    # Compute Möbius function up to K to identify square-free numbers.
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            for j in range(i2, K + 1, i2):
                mu[j] = 0.0

    # Domain for variables: square-free numbers in [2, K]
    sq_free = np.array([k for k in range(2, K + 1) if mu[k] != 0], dtype=np.int64)
    m = len(sq_free)
    
    # If no square-free numbers (unlikely for K>=2), return fallback
    if m == 0:
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values

    # Objective coefficients for linprog (minimizes c^T x)
    # We want to maximize -sum g(k) log(k)/k
    # So minimize sum g(k) log(k)/k
    c = np.array([np.log(k) / k for k in sq_free], dtype=np.float64)
    
    # Constraint matrix A_ub for M(x) <= 1
    # M(x) = sum_{k in D} g(k) (floor(x/k) - x/k)
    # A_ub[i, j] = (floor(x_i / k_j) - x_i / k_j)
    # b_ub[i] = 1
    
    x_vals = np.arange(1, 10 * K + 1)
    n_x = len(x_vals)
    
    # Construct A_ub efficiently using vectorized operations
    # A_ub has shape (n_x, m)
    # We can compute floor(x/k) - x/k for all x and k
    # x_vals[:, None] / sq_free[None, :]
    
    try:
        # Vectorized calculation
        ratios = x_vals[:, None] / sq_free[None, :]
        A_ub = np.floor(ratios) - ratios
    except MemoryError:
        # Fallback to loop if memory is an issue (unlikely for 2000x120)
        A_ub = np.zeros((n_x, m), dtype=np.float64)
        for i in range(n_x):
            x = x_vals[i]
            A_ub[i, :] = np.floor(x / sq_free) - x / sq_free

    b_ub = np.ones(n_x, dtype=np.float64)
    
    # Bounds: -10 <= g(k) <= 10
    bounds = [(-10.0, 10.0)] * m
    
    # Solve LP
    try:
        from scipy.optimize import linprog
    except ImportError:
        # If scipy is not available, use fallback
        domain = np.array([1, 2], dtype=np.int64)
        values = np.array([1.0, -2.0], dtype=np.float64)
        return domain, values
        
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    # Check time and result
    if result.success:
        g_vals = result.x
        
        # Compute f(1)
        # f(1) = -sum g(k)/k
        f1 = -np.sum(g_vals / sq_free)
        
        # Construct final domain and values
        # Domain includes 1 and sq_free
        domain = np.concatenate(([1], sq_free)).astype(np.int64)
        values = np.concatenate(([f1], g_vals)).astype(np.float64)
        
        # Verification Step
        # The evaluator clips values to [-10, 10] BEFORE adjusting f(1).
        # Our g_vals are already in [-10, 10].
        # f1 might be outside [-10, 10].
        # If f1 is clipped, the sum f(k)/k will no longer be 0, and the evaluator will re-adjust f(1).
        # However, the problem statement says:
        # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S..."
        # If f1 is clipped, S changes.
        # Let's simulate the evaluator's steps to be sure.
        
        # 1. Clip values
        clipped_values = np.clip(values, -10.0, 10.0)
        
        # 2. Compute S
        S = np.sum(clipped_values / domain)
        
        # 3. Adjust f(1)
        # Index of 1 in domain is 0
        clipped_values[0] = clipped_values[0] - S
        
        # 4. Check M(x) <= 1
        K_final = int(domain.max())
        x_check = np.arange(1, 10 * K_final + 1)
        
        # M(x) = sum f(k) floor(x/k)
        # We can compute this efficiently
        # M(x) = f(1)*x + sum_{k in sq_free} f(k) floor(x/k)
        
        # Let's just do the full sum to be safe, it's fast for small K
        M_max = -np.inf
        for x in x_check:
            # M(x) = sum_{k in domain} f(k) floor(x/k)
            # Using vectorized numpy
            # floor(x / domain)
            floors = np.floor(x / domain)
            Mx = np.dot(clipped_values, floors)
            if Mx > M_max:
                M_max = Mx
                
        if M_max <= 1.0001:
            # Compute objective a
            # a = -sum f(k) log(k) / k
            # Note: log(1)=0, so f(1) term is 0.
            log_domain = np.log(domain)
            # Avoid log(0) or log(negative), but domain is positive integers.
            # f(k)/k
            term = clipped_values * log_domain / domain
            a = -np.sum(term)
            
            return domain, values
        else:
            # Infeasible after clipping simulation?
            # This can happen if f1 was clipped.
            # If f1 > 10, it becomes 10. S was calculated with f1=10.
            # The adjustment f(1) = 10 - S might result in a different value.
            # Actually, if f1 is clipped, the "sum f(k)/k = 0" condition is enforced by the evaluator's adjustment.
            # My simulation above does exactly that.
            # If it's infeasible, we should return a known feasible solution.
            pass

    # Fallback: Use a simple known feasible solution
    # f(1)=1, f(2)=-2
    domain = np.array([1, 2], dtype=np.int64)
    values = np.array([1.0, -2.0], dtype=np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
