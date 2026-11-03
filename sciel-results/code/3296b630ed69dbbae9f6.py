import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the schoolbook decomposition as a guaranteed valid fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using a structured Strassen-like approach
    # For 2x5x6, we can potentially use block decomposition with Strassen's algorithm
    # on the 2x2 sub-blocks
    
    # The key insight: if we split the 5 columns into 3+2 and 6 rows into 3+3,
    # we can use Strassen on the 2x2 blocks
    # A (2x5) = [A_left (2x3) | A_right (2x2)]
    # B (5x6) = [B_top (3x6); B_bottom (2x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_right * B_bottom (2x2 * 2x6), we can use a variant of Strassen
    # But the standard Strassen applies to square matrices
    
    # Alternative approach: use the known result that the rank of the 2x2x2 
    # multiplication tensor is 7 (Strassen). We can extend this.
    
    # Let's try a direct approach: search for rank < 60 using alternating least squares
    # with multiple random initializations and structured initializations
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s limit
    
    # Target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs T."""
        T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=tol)
    
    def try_round_and_verify(U, V, W):
        """Try rounding to integers/half-integers and verify."""
        for scale in [1, 2]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None, None, None
    
    # Try ranks from 45 to 59 (Strassen-inspired lower bounds)
    # For 2x5x6, a reasonable lower bound is around 45-50
    
    for r in range(45, min(rank_schoolbook, 60)):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        
        # Try multiple random initializations
        max_restarts = 30 if r < 55 else 15
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization with some structure
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize to prevent numerical issues
            for c in range(r):
                norm = np.linalg.norm(U[:, c])
                if norm > 1e-10:
                    U[:, c] /= norm
                    V[:, c] *= norm
                norm = np.linalg.norm(V[:, c])
                if norm > 1e-10:
                    V[:, c] /= norm
                    W[:, c] *= norm
            
            converged = False
            best_err = float('inf')
            
            for it in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: solve for U given V, W
                # T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
                # For fixed i: T[i, :, :] = sum_c U[i,c] * outer(V[:,c], W[:,c])
                
                # Build the basis matrix for U
                # A_U has shape (m*p*p*n, r)
                # A_U[flat_idx, c] = V[j,c] * W[k,c] where flat_idx = j*(p*n) + k
                A_U = np.einsum('jc,kc->jkc', V, W).reshape(m*p*p*n, r)
                T_flat = T.reshape(n*m, m*p*p*n)
                
                try:
                    U_new = np.linalg.lstsq(A_U, T_flat.T, rcond=None)[0].T
                    U = U_new
                except:
                    pass
                
                # Update V
                A_V = np.einsum('ic,kc->ikc', U, W).reshape(n*m*p*n, r)
                T_perm_V = np.transpose(T, (1, 0, 2))
                T_flat_V = T_perm_V.reshape(m*p, n*m*p*n)
                
                try:
                    V_new = np.linalg.lstsq(A_V, T_flat_V.T, rcond=None)[0].T
                    V = V_new
                except:
                    pass
                
                # Update W
                A_W = np.einsum('ic,jc->ijc', U, V).reshape(n*m*m*p, r)
                T_perm_W = np.transpose(T, (2, 0, 1))
                T_flat_W = T_perm_W.reshape(p*n, n*m*m*p)
                
                try:
                    W_new = np.linalg.lstsq(A_W, T_flat_W.T, rcond=None)[0].T
                    W = W_new
                except:
                    pass
                
                # Check convergence
                T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.linalg.norm(T - T_rec)
                best_err = min(best_err, err)
                
                if err < 1e-10:
                    converged = True
                    break
                
                # Early stopping if error is not decreasing
                if it > 30 and err > best_err * 1.05:
                    break
            
            if converged:
                U_r, V_r, W_r = try_round_and_verify(U, V, W)
                if U_r is not None:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a solution at this rank, try lower ranks
            continue
    
    # Return the best result found
    if best_rank == rank_schoolbook:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
