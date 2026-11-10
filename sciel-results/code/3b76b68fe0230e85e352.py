# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
from scipy.optimize import linprog


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to:
      - sum_k f(k)/k == 0 (enforced by the evaluator via f(1) adjustment)
      - M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10*K]
    where K is the maximum key in the domain.

    Strategy:
      1. Formulate the problem as a linear program (LP).
      2. Use a moderate K (e.g., 200) to keep the LP size manageable for the HiGHS solver.
      3. The LP variables are f(1), ..., f(K).
      4. Objective: minimize sum (log(k)/k) * f(k)  (equivalent to maximizing a).
      5. Constraints:
         - Equality: sum f(k)/k = 0 (normalization handled by evaluator, but we include it to align with the 'sum_k f(k)/k = 0' condition in the objective calculation context, though the evaluator adjusts f(1) to enforce this. Actually, the evaluator enforces sum f(k)/k = 0 by adjusting f(1). If we include this constraint in our LP, we ensure our solution is already normalized, so the evaluator's adjustment will be 0 (or very small). This is safe and likely optimal because the objective term for k=1 is 0 (log(1)=0), so f(1) does not contribute to 'a'. Thus, the value of f(1) is determined solely by the constraint sum f(k)/k = 0. Therefore, we can fix f(1) = -sum_{k>1} f(k)/k. This reduces the number of variables by 1 and the equality constraint.
         - Inequality: M(x) <= 1.0001 for x = 1, ..., 10*K.
         - Bounds: -10 <= f(k) <= 10.
      
      Let's refine the LP formulation:
      Variables: f(2), ..., f(K). (f(1) is determined by the constraint sum_{k=1}^K f(k)/k = 0 => f(1) = -sum_{k=2}^K f(k)/k).
      Note: The evaluator clips values to [-10, 10]. So we must ensure f(1) is within [-10, 10] as well.
      Objective: maximize a = -sum_{k=2}^K f(k) log(k)/k. (Term for k=1 is 0).
      Minimize: c^T y where y = [f(2), ..., f(K)] and c_k = log(k)/k.
      
      Constraints:
      1. M(x) <= 1.0001 for x in [1, 10K].
         M(x) = f(1) * floor(x/1) + sum_{k=2}^K f(k) * floor(x/k)
              = (-sum_{j=2}^K f(j)/j) * x + sum_{k=2}^K f(k) * floor(x/k)
              = sum_{k=2}^K f(k) * (floor(x/k) - x/k)
              
         So the constraint becomes:
         sum_{k=2}^K f(k) * (floor(x/k) - x/k) <= 1.0001 for all x in [1, 10K].
         
      2. Bounds: -10 <= f(k) <= 10 for k=2..K.
      3. Bound on f(1): -10 <= -sum_{k=2}^K f(k)/k <= 10  =>  -10 <= sum_{k=2}^K f(k)/k <= 10.
         This is a linear constraint on y.
         Let v_k = 1/k. Then -10 <= sum v_k f(k) <= 10.
         This splits into two inequalities:
         sum (1/k) f(k) <= 10
         sum (-1/k) f(k) <= 10

    """
    t0 = time.time()
    deadline = t0 + 590.0  # 10s safety margin

    # Try a few values of K to see if we can improve, but K=200 is a good start.
    # The LP size grows with K^2 roughly (10K constraints x K variables).
    # For K=200: 2000 constraints x 199 variables. Very fast.
    # For K=500: 5000 constraints x 499 variables. Still fast.
    # For K=1000: 10000 constraints x 999 variables. Might be a bit slower but likely OK.
    
    best_score = -np.inf
    best_domain = None
    best_values = None

    # Fallback seed
    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # We will try K=200 first. If it succeeds and gives a good score, we might try larger K if time permits.
    # However, the objective a converges to 1. The LP with larger K should get closer to 1.
    # Let's just pick a K that balances time and accuracy. K=200 is safe. K=500 is likely safe.
    # Let's try K=200.

    K = 200

    # Variables: f(2), ..., f(K)
    n_vars = K - 1
    if n_vars <= 0:
        return fallback_domain, fallback_values

    k_vals = np.arange(2, K + 1)
    
    # Objective coefficients: c_k = log(k) / k
    c = np.log(k_vals) / k_vals

    # Constraints for M(x) <= 1.0001
    # M(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k)
    # We need this for x = 1, ..., 10*K
    
    max_x = 10 * K
    x_vals = np.arange(1, max_x + 1)
    
    # A_ub will be (max_x + 2) x n_vars
    # Rows 0..max_x-1: M(x) constraints
    # Row max_x: sum (1/k) f(k) <= 10
    # Row max_x+1: sum (-1/k) f(k) <= 10
    
    # Compute A_ub for M(x) constraints efficiently
    # A_ub[i, j] = floor((i+1) / (j+2)) - (i+1) / (j+2)
    # Note: k_vals[j] = j+2
    
    # Using broadcasting might be memory heavy for large K, but K=200 is small.
    # x: 1..2000, k: 2..200. 2000x199 matrix. ~400k floats. Fine.
    
    # Construct A_ub
    # x_col: shape (max_x, 1)
    # k_row: shape (1, n_vars)
    x_col = x_vals[:, np.newaxis].astype(np.float64)
    k_row = k_vals[np.newaxis, :].astype(np.float64)
    
    # floor(x/k)
    floor_part = (x_col // k_row).astype(np.float64)
    # x/k
    div_part = x_col / k_row
    
    A_M = floor_part - div_part
    
    # Bounds on f(1)
    # f(1) = -sum f(k)/k
    # -10 <= f(1) <= 10
    # -10 <= -sum f(k)/k => sum f(k)/k <= 10
    # 10 >= -sum f(k)/k => sum f(k)/k >= -10 => -sum f(k)/k <= 10
    
    v = 1.0 / k_vals
    A_f1_1 = v[np.newaxis, :]  # sum v_k f_k <= 10
    A_f1_2 = -v[np.newaxis, :] # sum -v_k f_k <= 10
    
    A_ub = np.vstack([A_M, A_f1_1, A_f1_2])
    b_ub = np.concatenate([np.full(max_x, 1.0001), [10.0, 10.0]])
    
    bounds = [(-10.0, 10.0)] * n_vars
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            f_vals = res.x
            # Construct full domain and values
            # f(1) = -sum f(k)/k
            f1 = -np.sum(f_vals / k_vals)
            
            # Check if f1 is within bounds (should be, but just in case)
            if f1 < -10.0 or f1 > 10.0:
                # This shouldn't happen due to constraints, but if it does, clip and re-check?
                # Or just return fallback.
                pass
            
            full_values = np.concatenate([[f1], f_vals])
            full_domain = np.arange(1, K + 1, dtype=np.int64)
            
            # Verify feasibility manually to be safe
            # M(x) = sum f(k) floor(x/k)
            # We can compute this quickly for a few x or all x.
            # Given the LP constraint was M(x) <= 1.0001, it should hold.
            # Let's compute the objective a
            # a = -sum f(k) log(k) / k
            # Note: log(1)=0, so term 1 is 0.
            a = -np.sum(full_values * np.log(full_domain.astype(np.float64)) / full_domain.astype(np.float64))
            
            # The evaluator computes:
            # 1. Clip values to [-10, 10]
            # 2. S = sum f(k)/k
            # 3. f(1) -= S
            # 4. Check M(x) <= 1.0001
            # 5. a = -sum f(k) log(k) / k
            
            # Our solution already satisfies sum f(k)/k = 0 (by construction of f1).
            # So S = 0, and f(1) doesn't change.
            # Values are within [-10, 10] by bounds.
            # So the evaluator's a should match our calculated a.
            
            if a > best_score:
                best_score = a
                best_domain = full_domain
                best_values = full_values
                
        else:
            # LP failed
            pass
            
    except Exception as e:
        # Fallback
        pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        return fallback_domain, fallback_values

# EVOLVE-BLOCK-END
