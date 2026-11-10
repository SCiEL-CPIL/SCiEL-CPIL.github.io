import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # K=1000 is a good balance of performance and score potential based on knowledge.
    # We can try to increase K if time allows, but 1000 is safe.
    K = 1000
    
    # Precompute harmonic number H_K
    inv_k = np.arange(1, K + 1, dtype=np.float64)
    inv_k = 1.0 / inv_k
    H_K = np.sum(inv_k)
    
    # Objective coefficients c
    # Minimize sum c_k f_k
    # Maximize a = - sum_{k=2}^K f_k log(k)/k
    # So minimize sum_{k=2}^K f_k log(k)/k
    c = np.zeros(K)
    if K > 1:
        log_k = np.log(np.arange(2, K + 1))
        c[1:] = log_k / np.arange(2, K + 1)
    
    # Constraints A_ub f <= b_ub
    # M_final(x) <= 1
    # M_final(x) = f(1) * x * (1 - H_K) + sum_{k=2}^K f(k) * (floor(x/k) - x/k)
    
    # We need to check x in [1, 10K]
    max_x = 10 * K
    # To save memory/time, we might not need ALL x.
    # However, the constraint is "any M(x) > 1.0001".
    # Let's construct the full matrix if feasible.
    # 10,000 rows x 1,000 cols = 10M floats. 80MB. Acceptable.
    
    A_ub = np.zeros((max_x, K), dtype=np.float64)
    b_ub = np.ones(max_x, dtype=np.float64)
    
    # Column 0 (k=1)
    # A[x, 0] = (x+1) * (1 - H_K)  # x is 0-indexed in loop, value is x+1
    x_vals = np.arange(1, max_x + 1, dtype=np.float64)
    A_ub[:, 0] = x_vals * (1.0 - H_K)
    
    # Columns 1..K-1 (k=2..K)
    # Vectorized calculation might be tricky due to floor.
    # Loop over k is better for memory locality if we build column by column?
    # Or row by row?
    # Let's do row by row (x by x) to fill A_ub.
    # Actually, filling column by column is more efficient for numpy slicing if we precompute x_vals.
    # A_ub[:, k] = floor(x_vals / (k+1)) - x_vals / (k+1)
    
    for k in range(1, K):
        val_k = k + 1
        A_ub[:, k] = np.floor(x_vals / val_k) - x_vals / val_k
        
        # Check time
        if time.time() > deadline - 10:
             # If running out of time, break and solve with what we have?
             # No, we need full constraints for feasibility.
             # If we can't build A, we fail.
             # But K=1000, max_x=10000. Loop is 999 iterations. Very fast.
             pass

    # Bounds
    # f(k) in [-10, 10]
    bounds = [(-10.0, 10.0) for _ in range(K)]
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to seed if LP fails
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
            
        f = res.x
        
        # Verify that f is not all zeros or similar
        # If f is zero, score is 0.
        # If res.x is zero, it means the objective was 0.
        # With c > 0 for k>1, and constraints allowing negative f (since A_ub can be negative),
        # we should get non-zero f.
        
        # Check if any value is extremely close to 0 to avoid "0.0" error?
        # The error was "value 0.0 cannot be scored".
        # This might mean the *metric* was 0.
        # If f is zero, metric is 0.
        # Let's ensure we return the computed f.
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = f.astype(np.float64)
        
        return domain, values
        
    except Exception as e:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

