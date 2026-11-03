import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Pre-compute index mappings for faster verification
    # For each (i,j,k), we need U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Let's precompute the row indices
    idx_U = np.zeros((n, m, p), dtype=int)
    idx_V = np.zeros((n, m, p), dtype=int)
    idx_W = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U[i, j, k] = i * m + j
                idx_V[i, j, k] = j * p + k
                idx_W[i, j, k] = k * n + i

    def verify(U, V, W, tol=1e-9):
        """Verify if U, V, W form a valid decomposition with half-integer entries."""
        # Check half-integer constraint
        if not np.all(np.isclose(U * 2, np.round(U * 2), atol=tol)):
            return False
        if not np.all(np.isclose(V * 2, np.round(V * 2), atol=tol)):
            return False
        if not np.all(np.isclose(W * 2, np.round(W * 2), atol=tol)):
            return False
        
        # Exact tensor check using einsum
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We can compute this more efficiently
        R = U.shape[1]
        # Compute for each (i,j,k)
        errors = np.zeros((n, m, p))
        for c in range(R):
            u_c = U[:, c]  # (nm,)
            v_c = V[:, c]  # (mp,)
            w_c = W[:, c]  # (pn,)
            # For each (i,j,k): u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            for i in range(n):
                for j in range(m):
                    u_val = u_c[i*m+j]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        errors[i, j, k] += u_val * v_c[j*p+k] * w_c[k*n+i]
        
        if not np.all(np.isclose(errors, 1.0, atol=tol)):
            return False
        return True

    def solve_for_W(U, V, R):
        """Solve linear system for W given U and V."""
        num_eqs = n * m * p
        num_vars = pn * R
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            var_idx = (k*n + i) * R + c
                            A[idx, var_idx] += coeff
                    idx += 1
        
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        W = W_flat.reshape((pn, R))
        return W

    def local_refine(U, V, W, max_iters=100):
        """Try to refine W by adjusting individual entries to nearest half-integers."""
        R = U.shape[1]
        pn = W.shape[0]
        
        # Compute current errors
        def compute_errors(W):
            errors = np.zeros((n, m, p))
            for c in range(R):
                u_c = U[:, c]
                v_c = V[:, c]
                w_c = W[:, c]
                for i in range(n):
                    for j in range(m):
                        u_val = u_c[i*m+j]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            errors[i, j, k] += u_val * v_c[j*p+k] * w_c[k*n+i]
            return errors - 1.0
        
        W_work = W.copy()
        prev_err = np.max(np.abs(compute_errors(W_work)))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            errors = compute_errors(W_work)
            if np.max(np.abs(errors)) < 1e-12:
                break
            
            # Find the worst error
            max_err_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
            i, j, k = max_err_idx
            err_val = errors[i, j, k]
            
            # Try adjusting W entries that affect this (i,j,k)
            # W[k*n+i, c] is affected by U[i*m+j, c] * V[j*p+k, c]
            improved = False
            for c in range(R):
                coeff = U[i*m+j, c] * V[j*p+k, c]
                if coeff == 0:
                    continue
                
                # Current W value
                w_curr = W_work[k*n+i, c]
                # Target: reduce error by err_val / coeff
                # So new W should be w_curr - err_val / coeff
                w_target = w_curr - err_val / coeff
                
                # Try rounding to nearest half-integer
                w_rounded = np.round(w_target * 2) / 2
                if w_rounded != w_curr:
                    W_work[k*n+i, c] = w_rounded
                    new_err = compute_errors(W_work)
                    if np.max(np.abs(new_err)) < prev_err:
                        prev_err = np.max(np.abs(new_err))
                        improved = True
                    else:
                        W_work[k*n+i, c] = w_curr
            
            if not improved:
                break
        
        return W_work

    # Search for lower ranks
    # Try ranks from 20 to 44
    for R in range(20, 45):
        if time.time() > deadline:
            break
        
        # Skip if we've already found this rank or better
        if R >= best_rank:
            continue
        
        # Limit iterations per rank based on time
        max_iters = 200
        iters_done = 0
        found = False
        
        while iters_done < max_iters and time.time() < deadline:
            iters_done += 1
            
            # Random U, V with sparse entries
            density = 0.3
            U = np.random.choice([-1, 0, 1], size=(nm, R), p=[0.15, 0.7, 0.15])
            V = np.random.choice([-1, 0, 1], size=(mp, R), p=[0.15, 0.7, 0.15])
            
            W = solve_for_W(U, V, R)
            if W is None:
                continue
            
            # Round W to nearest half-integer
            W_rounded = np.round(W * 2) / 2
            
            # Try local refinement
            W_refined = local_refine(U, V, W_rounded, max_iters=20)
            
            # Check if refined W is valid
            if verify(U, V, W_refined):
                best_rank = R
                best_U, best_V, best_W = U, V, W_refined
                found = True
                break
        
        # If found at this rank, we can continue to try lower ranks
        # The loop will naturally go to lower R next
    
    # Return best result
    if best_rank < rank_schoolbook:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
