"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import least_squares

def _validate(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]*B[j,k].
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W index: W[k*n + i, c] should equal sum_j U[i*m+j,c]*V[j*p+k,c].
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # For each (i,k): sum_j U3[i,j,:] * V3[j,k,:] should equal W3[k,i,:]
    # Wait, the tensor decomposition condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let's verify this condition directly as it's the definition of the rank.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(R, n, m, p, seed=42, time_limit=10.0):
    """Try to find a rank-R decomposition using scipy.optimize.least_squares."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Dimensions
    n_m = n * m
    m_p = m * p
    p_n = p * n
    total_vars = R * (n_m + m_p + p_n)
    
    # Precompute indices for fast evaluation
    # We need to map (i, j, k) to the indices in U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def residuals(x):
        # x is a flat vector of length total_vars
        # Split x into U, V, W
        U_flat = x[:R*n_m].reshape(n, m, R)
        V_flat = x[R*n_m:R*n_m + R*m_p].reshape(m, p, R)
        W_flat = x[R*n_m + R*m_p:].reshape(p, n, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c
                    # U[i,j,c] * V[j,k,c] * W[k,i,c]
                    s = np.sum(U_flat[i,j,:] * V_flat[j,k,:] * W_flat[k,i,:])
                    res[idx] = s - 1.0
                    idx += 1
        return res

    best_solution = None
    best_err = float('inf')
    
    # Multiple restarts
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize x
        x0 = np.zeros(total_vars)
        
        # Sparse initialization:
        # For each rank index c, pick a random subset of non-zero entries
        # This encourages sparsity which helps with integer solutions
        for c in range(R):
            # U part
            u_start = c * (n_m + m_p + p_n) # This is wrong if we split by matrix first
            # My split was: all U, then all V, then all W.
            # Let's stick to the split: U (R*n_m), V (R*m_p), W (R*p_n)
            # U index for (i,j,c) is (i*m+j)*R + c ? No, reshape was (n,m,R).
            # If x is flat, and we reshape to (n,m,R), the index is ((i*m)+j)*R + c.
            # Let's re-verify the reshape order.
            # np.array(x).reshape(n, m, R)
            # Index [i,j,c] is at flat index ((i*m) + j)*R + c.
            
            # So for a specific c, the non-zero entries in U are at indices:
            # ((i*m)+j)*R + c for chosen (i,j).
            
            # Let's just generate random sparse vectors for U, V, W parts
            pass # Logic below
            
        # Generate x0
        x0 = np.zeros(total_vars)
        # U part: indices 0 to R*n_m
        # V part: indices R*n_m to R*n_m + R*m_p
        # W part: indices R*n_m + R*m_p to end
        
        # For each c, we want to activate a few (i,j) in U, (j,k) in V, (k,i) in W
        for c in range(R):
            # Pick random i, j, k structure
            # Simple heuristic: pick a random i, j, k and set the corresponding entries to 1
            i_c = rng.randint(n)
            j_c = rng.randint(m)
            k_c = rng.randint(p)
            
            # U[i_c, j_c, c] = 1
            idx_u = ((i_c * m) + j_c) * R + c
            x0[idx_u] = rng.choice([-1, 1])
            
            # V[j_c, k_c, c] = 1
            idx_v = R*n_m + ((j_c * p) + k_c) * R + c
            x0[idx_v] = rng.choice([-1, 1])
            
            # W[k_c, i_c, c] = 1
            idx_w = R*n_m + R*m_p + ((k_c * n) + i_c) * R + c
            x0[idx_w] = rng.choice([-1, 1])
            
            # Add some noise to allow optimization to move away from discrete values
            # Or keep them 0/1 and let optimizer adjust?
            # Least squares works better with continuous init.
            # Let's add small random values to a subset of other entries
            # Actually, the above creates a rank-1 term structure.
            # Sum of R such terms = Rank R.
            # This is a valid starting point (a valid decomposition of rank R, but the sum of terms is R, not 1.
            # Wait, if we sum R terms, each contributing 1, the total is R.
            # We need the sum to be 1.
            # So we should scale them by 1/R?
            # Or just let the optimizer scale them.
            # If we initialize with 1s, the residual is R-1.
            # Optimizer will try to reduce the magnitude.
            
        # Run optimization
        try:
            sol = least_squares(residuals, x0, method='trf', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_solution = sol.x
        except Exception:
            continue

    if best_solution is not None and best_err < 1e-8:
        # Extract U, V, W
        U_flat = best_solution[:R*n_m].reshape(n, m, R)
        V_flat = best_solution[R*n_m:R*n_m + R*m_p].reshape(m, p, R)
        W_flat = best_solution[R*n_m + R*m_p:].reshape(p, n, R)
        
        # Round to integers or half-integers
        # The problem allows integer/half-integer entries.
        # Let's try rounding to nearest integer first.
        U_int = np.round(U_flat).astype(int)
        V_int = np.round(V_flat).astype(int)
        W_int = np.round(W_flat).astype(int)
        
        # Verify
        if _validate(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        # Try rounding to half-integers (multiply by 2, round, divide by 2)
        U_half = np.round(U_flat * 2) / 2.0
        V_half = np.round(V_flat * 2) / 2.0
        W_half = np.round(W_flat * 2) / 2.0
        
        # Convert to list of lists
        U_list = U_half.tolist()
        V_list = V_half.tolist()
        W_list = W_half.tolist()
        
        if _validate(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound up to schoolbook
    schoolbook_rank = n * m * p
    
    for R in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Allocate more time for lower ranks as they are more likely to be the answer
        # and harder to find? Or just fixed time per rank?
        # Let's give 10s to each rank attempt.
        time_limit = min(10.0, deadline - time.time() - 2)
        if time_limit <= 0:
            break
            
        result = _try_rank(R, n, m, p, seed=R, time_limit=time_limit)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
