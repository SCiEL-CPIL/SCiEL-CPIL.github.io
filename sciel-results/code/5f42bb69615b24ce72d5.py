import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    start_time = time.time()
    deadline = 150.0
    
    # Initialize schoolbook
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    current_rank = rank
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    # Helper to check if entries are close to half-integers
    def is_half_integer(arr, tol=1e-6):
        # 2 * arr should be close to integer
        doubled = 2.0 * arr
        closest = np.round(doubled)
        return np.allclose(doubled, closest, atol=tol)
        
    # Greedy reduction
    improved = True
    while improved and (time.time() - start_time) < deadline - 10.0:
        improved = False
        current_rank = U.shape[1]
        
        # Try to remove each column
        for c in range(current_rank):
            if (time.time() - start_time) > deadline - 10.0:
                break
                
            # Build system A * alpha = b
            # Variables: alpha for all other columns
            other_cols = [r for r in range(current_rank) if r != c]
            num_vars = len(other_cols)
            
            # Equations: for each (u, v, w) in the support of the tensor
            # The support is defined by the non-zero entries of the target tensor.
            # In our flattened representation, the target tensor T has 1s at:
            # u = i*m + j, v = j*p + k, w = k*n + i for all i, j, k.
            # There are n*m*p such equations.
            
            # We can build A and b efficiently.
            # A has shape (n*m*p, num_vars)
            # b has shape (n*m*p,)
            
            # To save time, we can sample or use the full system.
            # n*m*p = 120. num_vars ~ 119.
            # 120 x 119 system is small.
            
            A = np.zeros((n * m * p, num_vars))
            b = np.zeros(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        
                        # b part: contribution of column c
                        b[eq_idx] = U[u, c] * V[v, c] * W[w, c]
                        
                        # A part: contributions of other columns
                        for var_idx, r in enumerate(other_cols):
                            A[eq_idx, var_idx] = U[u, r] * V[v, r] * W[w, r]
                        
                        eq_idx += 1
            
            # Solve A * alpha = b
            # Use least squares if overdetermined
            try:
                alpha, residuals, rank_sys, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check if solution is valid (residuals small)
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
                
            # Check if the new entries are half-integers
            # The new column c is removed.
            # The other columns are updated: U[:, r] += alpha[var_idx] * U[:, c] ?
            # Wait. The equation is:
            # U_c V_c W_c = sum alpha_r U_r V_r W_r
            # This means we can replace the term c with the combination.
            # But we want to REMOVE c.
            # So we set U[:, c] = 0, etc.
            # And we update the OTHER columns?
            # No, the decomposition is a sum.
            # If we remove c, we must add the "deficit" to the other columns?
            # No, the equation says the term c IS EQUAL to the sum of others.
            # So we can simply drop c. The sum of others already accounts for it.
            # Wait. If T = sum_all terms.
            # And term_c = sum_others terms.
            # Then T = term_c + sum_others terms = 2 * sum_others terms.
            # This is WRONG.
            # The equation should be:
            # We want T = sum_{r != c} U'_r V'_r W'_r.
            # We know T = U_c V_c W_c + sum_{r != c} U_r V_r W_r.
            # If we can write U_c V_c W_c = sum_{r != c} \beta_r U_r V_r W_r,
            # Then T = sum_{r != c} (1 + \beta_r) U_r V_r W_r.
            # So we update the coefficients of the other columns.
            
            # So, new U[:, r] = U[:, r] + beta[var_idx] * U[:, c] ?
            # No. The term is U_r V_r W_r.
            # We are adding \beta_r * (U_c V_c W_c).
            # This is not a simple scaling of U_r.
            # U_c V_c W_c is a rank-1 tensor.
            # We cannot add a rank-1 tensor to U_r V_r W_r and keep it as U'_r V'_r W'_r.
            # UNLESS we can distribute the rank-1 tensor?
            # We can't.
            
            # So the logic "remove c" is flawed if we can't represent the sum of rank-1 tensors as a single rank-1 tensor.
            # The equation U_c V_c W_c = sum \alpha_r U_r V_r W_r is a tensor equality.
            # It does NOT imply that we can update U_r, V_r, W_r to absorb it.
            # It implies that the TENSOR term c is redundant IF the other terms are already present?
            # No. If T = T_c + T_others.
            # And T_c = T_others.
            # Then T = 2 T_others.
            # This changes the tensor.
            
            # So this approach is invalid for removing a column directly.
            
            # Correct approach for Strassen:
            # We define NEW tensors U', V', W'.
            # We don't just remove a column. We replace the set of columns with a new set.
            # For 2x2x2, we replace 8 columns with 7 new columns.
            # The new columns are linear combinations of the old ones?
            # No, the new columns are different vectors.
            
            # So we need to SEARCH for a new decomposition.
            # This is harder.
            
            pass # Skip this reduction logic as it's flawed for direct removal
            
    # If no reduction found, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
