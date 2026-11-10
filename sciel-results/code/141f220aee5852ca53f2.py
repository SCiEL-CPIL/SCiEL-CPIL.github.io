# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Fallback to seed result
    def fallback():
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # We want to maximize a = -sum_k f(k) * log(k) / k
    # Subject to M(x) <= 1.0001 for all x in [1, 10K]
    # And -10 <= f(k) <= 10

    # The objective function is linear in f.
    # The constraints M(x) <= 1.0001 are linear in f.
    # The bounds are linear.
    # This is a standard Linear Programming problem.

    # To handle the S constraint (f(1) = -S), we can eliminate f(1) from the variables.
    # Let f(1) be a variable too, or just include it.
    # Actually, the evaluator computes S = sum f(k)/k over clipped values.
    # Then f(1) is replaced by f(1) - S.
    # So effectively, the value at 1 is f(1) - sum_{k in domain} f(k)/k.
    # Let's just include f(1) in the domain and values, and let the LP optimize it.
    # Wait, the evaluator logic:
    # 1. Clip values to [-10, 10].
    # 2. Compute S = sum_k f(k)/k.
    # 3. f(1) = f(1) - S.
    # 4. Check sum_k f(k)/k = 0.
    # 5. M(x) = sum_k f(k) * floor(x/k).
    #
    # If we include 1 in the domain, let f(1) be a variable v_1.
    # The evaluator will set v_1' = v_1 - S.
    # The constraint is M(x) <= 1.0001.
    # M(x) = v_1' * x + sum_{k>1} v_k * floor(x/k).
    # This makes the constraint non-linear in v_1 if we treat v_1 as the input variable?
    # No, v_1' is linear in v_1 and all other v_k.
    # v_1' = v_1 - sum_{k>1} v_k/k - v_1/1 = - sum_{k>1} v_k/k.
    # So f(1) effectively becomes - sum_{k>1} v_k/k.
    # This means f(1) is determined by the other variables.
    # So we can eliminate f(1) from the decision variables.
    # Let the decision variables be v_k for k in D, where D is a subset of {2, ..., K}.
    # Then f(1) is fixed as -sum_{k in D} v_k/k.
    #
    # Objective: a = - ( f(1)*log(1)/1 + sum_{k in D} v_k * log(k) / k )
    # log(1) = 0, so the f(1) term vanishes in the objective!
    # a = - sum_{k in D} v_k * log(k) / k.
    #
    # Constraints:
    # M(x) = f(1)*x + sum_{k in D} v_k * floor(x/k) <= 1.0001 for x in [1, 10K].
    # f(1) = - sum_{j in D} v_j / j.
    # So M(x) = -x * sum_{j in D} v_j / j + sum_{k in D} v_k * floor(x/k).
    # M(x) = sum_{k in D} v_k * ( floor(x/k) - x/k ).
    #
    # This is a linear constraint in v_k.
    # Note that for k | x, floor(x/k) - x/k = 0.
    # For k not dividing x, floor(x/k) - x/k is negative (between -1 and 0).
    # Specifically, let x = q*k + r, 0 <= r < k.
    # floor(x/k) = q. x/k = q + r/k.
    # floor(x/k) - x/k = -r/k.
    # So the coefficient of v_k in M(x) is - (x mod k) / k.
    #
    # We want to maximize - sum c_k v_k where c_k = log(k)/k.
    # Since we want to maximize a, and the constraint is sum coeff * v <= 1.0001,
    # and coeff is negative (or zero), we should make v_k as large as possible (positive) to make the LHS small (more negative), allowing more "room"?
    # Wait.
    # Maximize - sum c_k v_k.
    # Constraint: sum_{k} (- (x mod k) / k) v_k <= 1.0001.
    # Let A_{x,k} = - (x mod k) / k.
    # A_{x,k} <= 0.
    # We want to maximize - sum c_k v_k.
    # This is equivalent to minimizing sum c_k v_k.
    #
    # If we set v_k to be positive, the term - sum c_k v_k becomes negative (since c_k > 0).
    # We want a to be as large as possible.
    # If we set v_k to be negative, - sum c_k v_k becomes positive.
    # So we want v_k to be negative.
    # Let v_k = -u_k, where u_k >= 0.
    # Maximize sum c_k u_k.
    # Constraint: sum A_{x,k} (-u_k) <= 1.0001 => - sum A_{x,k} u_k <= 1.0001.
    # Since A_{x,k} <= 0, let B_{x,k} = -A_{x,k} = (x mod k) / k >= 0.
    # Constraint: sum B_{x,k} u_k <= 1.0001.
    # Bounds: -10 <= v_k <= 10 => -10 <= -u_k <= 10 => -10 <= u_k <= 10.
    # Also u_k >= 0? No, v_k can be positive.
    # But intuitively, to maximize sum c_k u_k subject to sum B u <= 1, we want u_k to be as large as possible for indices with high c_k/B ratio.
    # If we restrict u_k >= 0, we are forcing v_k <= 0.
    # Is it ever optimal to have v_k > 0?
    # If v_k > 0, it contributes negatively to the objective and positively to the LHS of the constraint (making it tighter).
    # So it is always suboptimal to have v_k > 0.
    # Thus, we can assume v_k <= 0 for all k > 1.
    #
    # So we solve:
    # Maximize sum_{k in D} c_k u_k
    # Subject to:
    #   sum_{k in D} B_{x,k} u_k <= 1.0001 for all x in [1, 10K]
    #   0 <= u_k <= 10
    #
    # This is a standard LP.
    # Variables: u_k for k in D.
    # Objective: maximize c^T u.
    # Constraints: B u <= b, 0 <= u <= 10.
    #
    # We can use scipy.optimize.linprog.
    # linprog minimizes c^T x. So we minimize -c^T u.
    # c_lp = -c.
    # A_ub = B.
    # b_ub = 1.0001.
    # bounds = [(0, 10) for k in D].

    # We need to choose D.
    # The number of variables is |D|.
    # The number of constraints is 10K.
    # For K=100, 10K=1000 constraints.
    # If |D| is small, it's fast.
    # Let's try K=1000. 10K=10000 constraints.
    # D can be all numbers 2..K. |D| ~ 1000.
    # 1000 variables, 10000 constraints.
    # This might be slow for linprog if not careful.
    # However, B is sparse.
    # linprog in scipy supports sparse matrices.
    
    # Let's try a moderate K.
    # K=200. 10K=2000 constraints.
    # D = 2..200. |D|=199.
    # This is very manageable.

    # Let's try K=1000.
    # D = 2..1000. |D|=999.
    # 10000 constraints.
    # B is sparse.
    
    # We need to be careful with time.
    # If linprog takes too long, we fall back.

    def solve_lp(K):
        if time.time() > deadline:
            return None

        # D is 2..K
        D = np.arange(2, K + 1)
        n_vars = len(D)
        
        # c_k = log(k) / k
        c = np.log(D.astype(np.float64)) / D.astype(np.float64)
        
        # We want to maximize sum c_k u_k
        # So minimize -sum c_k u_k
        c_lp = -c
        
        # Construct B matrix (sparse)
        # B_{x, k} = (x mod k) / k for x in [1, 10K], k in D
        max_x = 10 * K
        n_cons = max_x
        
        # We can build the sparse matrix row by row or using lists of indices.
        # Since n_vars and n_cons are moderate, we can try to build it efficiently.
        # However, constructing a 10000x1000 sparse matrix might take some time.
        
        # Let's estimate the size.
        # For each k, B_{x,k} is non-zero for x not divisible by k.
        # Density is roughly 1 - 1/k.
        # So it's dense-ish.
        # 10000 * 1000 = 10 million entries.
        # This is fine for memory.
        
        # Let's construct B as a dense matrix if K is small enough, else sparse.
        # For K=1000, 10M floats is 80MB. Acceptable.
        # But constructing it in Python loop is slow.
        
        # Vectorized construction:
        # x = np.arange(1, max_x + 1)
        # k = D
        # We need (x mod k) / k.
        # Broadcasting: x[:, None] % k[None, :]
        # This creates a 10000x1000 array.
        
        x = np.arange(1, max_x + 1, dtype=np.float64)
        k = D.astype(np.float64)
        
        # This might be memory intensive for large K.
        # Let's stick to K=200 first to be safe and fast.
        # If K=200, 2000x199 array is small.
        
        if K > 500:
             # Use sparse construction if needed, but let's try dense for simplicity first
             # and rely on K being small enough.
             pass
        
        # Construct B
        # B[i, j] corresponds to x = i+1, k = D[j]
        # x_mod_k = x[:, None] % D[None, :]
        # B = x_mod_k / D[None, :]
        
        # To save memory/time, let's iterate over rows if K is large, but for K=200 it's fine.
        # Let's try K=200.
        
        x_mod_k = x[:, None] % D[None, :]
        B = x_mod_k / D[None, :]
        
        # Constraints: B u <= 1.0001
        b_ub = np.full(n_cons, 1.0001)
        
        # Bounds: 0 <= u_k <= 10
        bounds = [(0, 10) for _ in range(n_vars)]
        
        # Solve LP
        res = linprog(c_lp, A_ub=B, b_ub=b_ub, bounds=bounds, method='highs')
        
        if time.time() > deadline:
            return None
            
        if not res.success:
            return None
            
        u = res.x
        
        # Convert back to v
        v = -u
        
        # Check if any v is significantly non-zero
        if np.max(np.abs(v)) < 1e-6:
             return None
             
        # Construct domain and values
        # Domain includes 1?
        # The evaluator expects domain to hold distinct positive integers.
        # If we return domain = [1, 2, ..., K], values = [f(1), v_2, ..., v_K]
        # The evaluator will recompute f(1).
        # We should return f(1) = 0 or whatever, it gets replaced.
        # But to be clean, let's return the actual f(1) calculated.
        # f(1) = - sum v_k / k
        
        S = np.sum(v / D)
        f1 = -S
        
        domain = np.concatenate(([1], D))
        values = np.concatenate(([f1], v))
        
        # Verify constraints just in case
        # M(x) = f1 * x + sum v_k * floor(x/k)
        # We already enforced this in LP.
        
        return domain, values

    # Try different K values
    # Start with small K to get a valid solution quickly, then try larger if time permits.
    # The objective a scales with the number of terms and their magnitude.
    # Larger K allows more terms, potentially higher a.
    
    best_result = None
    best_score = -np.inf
    
    for K in [100, 200, 500, 1000]:
        if time.time() > deadline:
            break
            
        res = solve_lp(K)
        if res is None:
            continue
            
        domain, values = res
        
        # Calculate the score (metric a) to compare
        # a = -sum f(k) log(k) / k
        # f(1) term is 0.
        a = -np.sum(values * np.log(domain.astype(np.float64)) / domain)
        
        if a > best_score:
            best_score = a
            best_result = (domain, values)
            
        # If we have a good score, maybe stop?
        # Or continue to see if larger K is better.
        # Given the time limit, 1000 might be slow.
        # Let's check time.
        if time.time() > deadline - 60:
            break

    if best_result is not None:
        return best_result
        
    return fallback()

# EVOLVE-BLOCK-END
