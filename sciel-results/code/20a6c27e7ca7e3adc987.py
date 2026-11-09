"""Matrix multiplication 2x3x6 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_start = time.time()
DEADLINE = 110.0  # safety margin before 120s limit

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # Build the target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Lower bound on rank
    lower_bound = max(dim_u, dim_v, dim_w)  # 18
    
    # Try to find a decomposition with rank < 36
    # Use a structured approach: exploit the matrix multiplication structure
    
    # For 2x3x6, try to construct a decomposition with rank 24 or lower
    # Idea: use algebraic identities to reduce the number of multiplications
    
    # Let's try a specific construction:
    # Group by i (row of A) and k (col of B)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For fixed i and k, we need sum_j A[i,j] * B[j,k]
    # This is an inner product of two 3-vectors
    
    # We can try to use the fact that 3-vector inner products can be computed
    # with fewer than 3 multiplications using special identities
    
    # Actually, for 3-dimensional inner products, the minimal number of multiplications
    # is 3 (it's a bilinear map R^3 x R^3 -> R, and the rank is 3)
    
    # So for each (i,k) pair, we need at least 3 multiplications
    # Total: 2 * 6 * 3 = 36, which is the schoolbook
    
    # But we can share multiplications across different (i,k) pairs!
    # This is the key insight for Strassen-like algorithms
    
    # Let's try to find a decomposition with rank < 36 by sharing terms
    
    # Approach: set up the problem as finding vectors u_c, v_c, w_c
    # such that the tensor decomposition holds
    
    # For rank r, we have r * (6 + 18 + 12) = 36r unknowns
    # and 36 equations (one for each nonzero entry of T)
    
    # This is an underdetermined system for r >= 1, but the equations are nonlinear
    
    # Let's try a greedy approach: iteratively add rank-1 terms
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try ranks from lower_bound to 35
    for r in range(lower_bound, best_rank):
        if time.time() > DEADLINE:
            break
        
        # Try to find a decomposition of rank r using random restarts
        found = False
        
        for restart in range(5):
            if time.time() > DEADLINE or found:
                break
            
            # Initialize with random vectors
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            # Try to optimize: minimize ||sum_c u_c v_c w_c - T||^2
            # Use a simple gradient-free approach: coordinate descent on scales
            
            # First, fix the directions and optimize the scales
            # Let U = U_dir * diag(alpha), V = V_dir, W = W_dir
            # Then sum_c alpha_c * (U_dir[:,c] * V_dir[:,c] * W_dir[:,c]) = T
            
            # This is a linear system in alpha_c!
            # Build the matrix M where M[entry, c] = U_dir[i,c]*V_dir[j,c]*W_dir[k,c]
            # for the entry (i,j,k)
            
            # Flatten T to a vector
            T_flat = T.flatten()
            
            # Build M
            M = np.zeros((dim_u * dim_v * dim_w, r))
            for c in range(r):
                M[:, c] = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]).flatten()
            
            # Solve M @ alpha = T_flat (least squares)
            try:
                alpha, residuals, rank_m, sv = np.linalg.lstsq(M, T_flat, rcond=None)
            except:
                continue
            
            # Check the fit
            T_recon = (M @ alpha).reshape(dim_u, dim_v, dim_w)
            error = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
            
            if error < 1e-8:
                # We have a valid decomposition!
                # Now scale: U_final = U * diag(alpha), V_final = V, W_final = W
                U_final = U * alpha
                V_final = V
                W_final = W
                
                # Check if we can make them integer/half-integer
                # Try to find a common denominator
                found = True
                best_rank = r
                best_decomp = (U_final, V_final, W_final)
                break
        
        if found:
            break
    
    if best_decomp is not None:
        U_final, V_final, W_final = best_decomp
        
        # Try to convert to integer/half-integer
        # Check if multiplying by 2 gives integers
        for denom in [1, 2, 4, 8]:
            U_check = np.round(U_final * denom)
            V_check = np.round(V_final * denom)
            W_check = np.round(W_final * denom)
            
            # Verify the decomposition holds with the rounded values
            T_check = np.zeros((dim_u, dim_v, dim_w))
            for c in range(best_rank):
                T_check += np.einsum('i,j,k->ijk', U_check[:, c], V_check[:, c], W_check[:, c])
            
            if np.allclose(T_check, T * (denom ** 3), atol=1e-6):
                # Valid! Return the integer values
                return {
                    "U": U_check.tolist(),
                    "V": V_check.tolist(),
                    "W": W_check.tolist()
                }
        
        # If we can't make them integer, return as floats (but the problem requires integer/half-integer)
        # So we need to find a valid integer decomposition
    
    # Fallback: return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
