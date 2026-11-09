# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    # Choose K: maximum key
    K = 200000  # Max allowed
    
    # Variables: f(k) for k = 2, ..., K
    # We'll use a smaller K to keep LP tractable, then check feasibility
    # Start with moderate K
    K_trial = 5000
    
    # Objective: maximize -sum f(k)*log(k)/k
    # linprog minimizes, so minimize sum f(k)*log(k)/k
    # c[k-2] = log(k)/k for k = 2..K_trial
    ks = np.arange(2, K_trial + 1)
    c = np.log(ks) / ks  # minimize sum c_i * f_i
    
    # Constraints: for x in [1, 10*K_trial], sum_k f(k)*(floor(x/k) - x/k) <= 1.0001
    # Note: floor(x/k) - x/k is in (-1, 0] for all k, x
    # So these are all negative or zero coefficients
    
    # We need to build the constraint matrix A_ub where A_ub @ f <= b_ub
    # A_ub has shape (num_x, K_trial - 1)
    # This could be huge: 10*5000 = 50000 rows, 4999 cols
    
    # Let's be smarter: many x values give the same constraint pattern
    # floor(x/k) changes only when x crosses multiples of k
    # For each k, floor(x/k) - x/k = -frac(x/k)
    # The constraint is sum_k f(k) * (-frac(x/k)) <= 1.0001
    # i.e., -sum_k f(k)*frac(x/k) <= 1.0001
    # i.e., sum_k f(k)*frac(x/k) >= -1.0001
    
    # Let me reconsider the sign. floor(x/k) - x/k = -(x/k - floor(x/k)) = -frac(x/k)
    # So constraint: sum_k f(k)*(-frac(x/k)) <= 1.0001
    # => -sum_k f(k)*frac(x/k) <= 1.0001
    # => sum_k f(k)*frac(x/k) >= -1.0001
    
    # For linprog: A_ub @ f <= b_ub
    # We need: -sum_k f(k)*frac(x/k) <= 1.0001
    # So A_ub[i, j] = -frac(x_i / k_j) for constraint i, variable j
    
    # But 50000 x 4999 matrix is ~250M entries, too large
    # Let's use a smaller K and fewer x values (only critical ones)
    
    # Actually, let's use K=500 and check
    K_trial = 500
    ks = np.arange(2, K_trial + 1)
    c = np.log(ks) / ks
    
    # For constraints, we only need to check x values where the constraint is tight
    # The most binding constraints are likely at small x or where frac(x/k) is large
    # Let's sample x values
    x_max = 10 * K_trial
    # Check all x from 1 to x_max, but that's 5000 x 499 matrix = 2.5M entries, manageable
    xs = np.arange(1, x_max + 1)
    num_x = len(xs)
    num_vars = len(ks)
    
    # Build A_ub: A_ub[i, j] = -frac(xs[i] / ks[j])
    # Use vectorized computation
    # frac(x/k) = x/k - floor(x/k)
    X = xs[:, np.newaxis].astype(np.float64)  # (num_x, 1)
    Ks = ks[np.newaxis, :].astype(np.float64)  # (1, num_vars)
    ratio = X / Ks  # (num_x, num_vars)
    frac = ratio - np.floor(ratio)
    A_ub = -frac  # (num_x, num_vars)
    
    b_ub = np.ones(num_x) * 1.0001
    
    # Bounds: -10 <= f(k) <= 10
    bounds = [(-10, 10)] * num_vars
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    if result.success:
        f_vals = result.x  # f(k) for k = 2..K_trial
        # Clip to [-10, 10] for safety
        f_vals = np.clip(f_vals, -10, 10)
        
        # Build domain and values
        domain = np.concatenate(([1], ks))
        # f(1) can be anything since it cancels; set to 0
        values = np.concatenate(([0.0], f_vals))
        
        # Verify feasibility
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            # Quick check
            S = np.sum(values / domain)
            f_adj = values.copy()
            f_adj[0] -= S  # f(1) adjustment
            
            # Check M(x) for some x
            K_max = max(domain)
            x_check = np.arange(1, min(10 * K_max, 10000) + 1)
            # Compute M(x) = sum_k f_adj(k) * floor(x/k)
            # This is expensive, do sparse check
            max_M = 0
            for x in x_check:
                M = np.sum(f_adj * (x // domain))
                if M > max_M:
                    max_M = M
            if max_M > 1.0001:
                # Infeasible, fallback
                domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
                values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
                return domain, values
        
        return domain, values
    else:
        # LP failed, fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
