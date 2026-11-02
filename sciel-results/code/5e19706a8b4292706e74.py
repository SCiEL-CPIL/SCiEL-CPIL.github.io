"""Matrix multiplication 3x3 x 3x7 - rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # In the W convention: W index is k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    def schoolbook():
        """Schoolbook decomposition, rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    best_rank = rank_sb
    
    def verify(U, V, W):
        """Verify that T = sum_r u_r ⊗ v_r ⊗ w_r."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=1e-9, rtol=1e-9)
    
    def round_half_integer(U, V, W, tol=1e-6):
        """Round entries to nearest half-integer and verify."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None
    
    # Lower bound for rank
    lb = max(n*m, m*p, p*n, n*p)  # = max(9, 21, 21, 21) = 21
    
    def try_rank_via_als(r, max_iter=300, n_attempts=5, seed=42):
        """Try to find a rank-r decomposition via alternating least squares."""
        rng = np.random.RandomState(seed)
        
        # Precompute unfoldings
        T1 = T.reshape(n*m, m*p * p*n)
        T2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
        T3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with different scales
            scale = 1.0 + attempt * 0.5
            U = rng.randn(n*m, r) * scale
            V = rng.randn(m*p, r) * scale
            W = rng.randn(p*n, r) * scale
            
            success = False
            prev_err = np.inf
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Update U: minimize ||T1 - U @ M||_F where M[(b,c), r] = V[b,r]*W[c,r]
                M = np.zeros((m*p * p*n, r))
                for b in range(m*p):
                    for c in range(p*n):
                        M[b * p*n + c, :] = V[b, :] * W[c, :]
                try:
                    U_new = T1 @ np.linalg.lstsq(M, T1.T, rcond=None)[0].T
                except:
                    U_new = U
                
                # Update V: minimize ||T2 - V @ M2||_F where M2[(a,c), r] = U[a,r]*W[c,r]
                M2 = np.zeros((n*m * p*n, r))
                for a in range(n*m):
                    for c in range(p*n):
                        M2[a * p*n + c, :] = U[a, :] * W[c, :]
                try:
                    V_new = T2 @ np.linalg.lstsq(M2, T2.T, rcond=None)[0].T
                except:
                    V_new = V
                
                # Update W: minimize ||T3 - W @ M3||_F where M3[(a,b), r] = U[a,r]*V[b,r]
                M3 = np.zeros((n*m * m*p, r))
                for a in range(n*m):
                    for b in range(m*p):
                        M3[a * m*p + b, :] = U[a, :] * V[b, :]
                try:
                    W_new = T3 @ np.linalg.lstsq(M3, T3.T, rcond=None)[0].T
                except:
                    W_new = W
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = np.linalg.norm(T_recon - T)
                
                if err < 1e-8:
                    success = True
                    break
                
                if abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            if success:
                rounded = round_half_integer(U, V, W)
                if rounded:
                    U_r, V_r, W_r = rounded
                    return U_r, V_r, W_r, r
                return U, V, W, r
        
        return None
    
    # Try ranks from lower bound upward
    # Be strategic: try the most likely ranks first
    rank_candidates = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40, 45, 50, 55, 60]
    
    for r in rank_candidates:
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            break
        try:
            result = try_rank_via_als(r, max_iter=400, n_attempts=5, seed=100 + r)
            if result:
                U_r, V_r, W_r, rank_r = result
                if verify(U_r, V_r, W_r):
                    best_result = (U_r, V_r, W_r, rank_r)
                    best_rank = rank_r
                    # If we found a good rank, we can stop or try to improve
                    if rank_r <= 23:
                        break
        except Exception:
            continue
    
    # Convert best result to list-of-lists format
    U_best, V_best, W_best, rank_best = best_result
    U_out = U_best.tolist()
    V_out = V_best.tolist()
    W_out = W_best.tolist()
    
    # Clean up floating point to half-integers
    def clean_half_int(matrix):
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                val2 = round(val * 2) / 2.0
                if val2 == int(val2):
                    new_row.append(int(val2))
                else:
                    new_row.append(val2)
            result.append(new_row)
        return result
    
    U_out = clean_half_int(U_out)
    V_out = clean_half_int(V_out)
    W_out = clean_half_int(W_out)
    
    # Final verification with cleaned values
    U_final = np.array(U_out, dtype=np.float64)
    V_final = np.array(V_out, dtype=np.float64)
    W_final = np.array(W_out, dtype=np.float64)
    
    if not verify(U_final, V_final, W_final):
        # Fallback to schoolbook if cleaned version doesn't verify
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()
        U_out = clean_half_int(U_out)
        V_out = clean_half_int(V_out)
        W_out = clean_half_int(W_out)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
