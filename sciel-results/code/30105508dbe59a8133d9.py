"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # n=2, m=4, p=6
    # U: 8 x r
    # V: 24 x r
    # W: 12 x r
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    # Construct the tensor T where T[i, j, k] = 1 if this term exists in matmul
    # Actually, the tensor for matrix multiplication A(nxm) * B(mxp) -> C(nxp)
    # is defined such that C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T has indices (i, j, k) and T[i, j, k] = 1 for all valid i, j, k.
    # Wait, the standard tensor for matmul T_{ijk} is 1 if we are computing C_ik from A_ij and B_jk.
    # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # We want to find U, V, W of rank r such that:
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: The problem statement says W[k*n+i, r].
    # Let's reshape T to (n*m, m*p, p*n) to match the matrix shapes?
    # No, the tensor is naturally 3D.
    # U is (n*m, r). Index (i, j) -> i*m + j.
    # V is (m*p, r). Index (j, k) -> j*p + k.
    # W is (p*n, r). Index (k, i) -> k*n + i.
    
    # Let's use ALS to minimize the reconstruction error for a given rank r.
    # Error = || T - sum_r U_r V_r W_r ||^2
    
    def try_rank(r, num_restarts=5, max_iters=100):
        """
        Attempt to find a rank-r decomposition.
        Returns (U, V, W) as lists of lists of Fractions if successful, else None.
        """
        if time.time() > deadline:
            return None
            
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Random initialization
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Normalize to prevent divergence
            for col in range(r):
                norm_U = np.linalg.norm(U[:, col])
                norm_V = np.linalg.norm(V[:, col])
                norm_W = np.linalg.norm(W[:, col])
                if norm_U < 1e-6 or norm_V < 1e-6 or norm_W < 1e-6:
                    continue
                # Scale so that product of norms is 1? Or just standardize.
                # Let's just update one factor at a time.
                
            err = float('inf')
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Update U
                # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                # For fixed r, U[i, j, r] = sum_k T[i, j, k] V[j, k, r] W[k, i, r] / (sum_k V[j, k, r] W[k, i, r] * V[j, k, r] W[k, i, r])?
                # No, this is not linear regression in that simple form because V and W are coupled.
                # Let's reshape.
                # Let M_U be (n*m) x (m*p*p*n)? No.
                
                # Alternative: Use gradient descent or Gauss-Newton.
                # Or, fix V and W, solve for U via least squares.
                # T_flat[i, j, k] = sum_r U[i, j, r] * (V[j, k, r] * W[k, i, r])
                # Let Z_U[i, j, r] = V[j, k, r] * W[k, i, r] summed over k? No, k is an index of T.
                
                # Let's create matrices for the linear system.
                # For U:
                # We want to minimize sum_{i,j,k} (T[i,j,k] - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
                # This is not a simple linear system for U because V and W are also variables.
                # But if we fix V and W, it IS linear in U.
                # Let's define for each (i, j):
                # Residual[i, j, k] = T[i, j, k] - sum_{r'} U[i, j, r'] V[j, k, r'] W[k, i, r']
                # We want to find U[i, j, :] that minimizes sum_k Residual^2.
                # This is a linear least squares problem for U[i, j, :].
                # Let A_U[i, j, r, k] = V[j, k, r] * W[k, i, r].
                # Then T[i, j, k] approx sum_r A_U[i, j, r, k] U[i, j, r].
                
                # Similarly for V and W.
                
                # To speed up, we can use Numpy broadcasting.
                
                # Update U
                # A_U shape: (n, m, r, p) -> V[j,k,r] * W[k,i,r]
                # V is (m, p, r). W is (p, n, r).
                # We need V[j, k, r] * W[k, i, r].
                # Let's transpose V to (m, r, p) and W to (n, r, p)?
                # V[j, k, r] -> V_T[j, r, k]
                # W[k, i, r] -> W_T[i, r, k]
                # Product: sum_k? No, k is the free index in the residual.
                # We want a matrix M such that M[i, j, k, r] = V[j, k, r] * W[k, i, r].
                # Then T[i, j, k] = sum_r M[i, j, k, r] U[i, j, r].
                
                # Let's construct M_U efficiently.
                # V: (m, p, r)
                # W: (p, n, r)
                # We want M[i, j, k, r] = V[j, k, r] * W[k, i, r]
                # This is an outer product-like operation.
                
                # Reshape V to (m, 1, p, 1, r)? No.
                # V[j, k, r] depends on j, k, r.
                # W[k, i, r] depends on k, i, r.
                
                # Let's use einsum.
                # M_U[i, j, k, r] = V[j, k, r] * W[k, i, r]
                # We can compute this as:
                # Temp_V = V.T # (r, m, p)
                # Temp_W = W.T # (r, p, n)
                # This is getting complicated. Let's just loop over i, j for U update.
                # n*m is 8. That's small.
                
                for i in range(n):
                    for j in range(m):
                        # We need to solve for U[i, j, :]
                        # Minimize sum_k (T[i, j, k] - sum_r U[i, j, r] V[j, k, r] W[k, i, r])^2
                        # Let b[i, j, k] = T[i, j, k]
                        # Let A_mat[k, r] = V[j, k, r] * W[k, i, r]
                        # We want U_vec[r] such that A_mat @ U_vec approx b_vec.
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            for r_idx in range(r):
                                A_mat[k, r_idx] = V[j, k, r_idx] * W[k, i, r_idx]
                        b_vec = T[i, j, :]
                        # Solve least squares
                        try:
                            U_new, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                            U[i, j, :] = U_new
                        except:
                            pass

                # Update V
                # Minimize sum_{i,k} (T[i, j, k] - sum_r U[i, j, r] V[j, k, r] W[k, i, r])^2
                # For fixed j:
                # A_mat[i, k, r] = U[i, j, r] * W[k, i, r]
                # We want V[j, k, r] such that sum_r A_mat[i, k, r] V[j, k, r] approx T[i, j, k]
                # This is a bit tricky because V[j, k, r] is multiplied by W[k, i, r].
                # Let's group by k.
                # For each k, we have a system for V[j, k, :].
                # Minimize sum_i (T[i, j, k] - sum_r U[i, j, r] V[j, k, r] W[k, i, r])^2
                # Let A_mat2[i, r] = U[i, j, r] * W[k, i, r]
                # We want V_vec[r] such that A_mat2 @ V_vec approx T[:, j, k]
                
                for j in range(m):
                    for k in range(p):
                        A_mat2 = np.zeros((n, r))
                        for i in range(n):
                            for r_idx in range(r):
                                A_mat2[i, r_idx] = U[i, j, r_idx] * W[k, i, r_idx]
                        b_vec2 = T[:, j, k]
                        try:
                            V_new, residuals, rank, sv = np.linalg.lstsq(A_mat2, b_vec2, rcond=None)
                            V[j, k, :] = V_new
                        except:
                            pass

                # Update W
                # Minimize sum_{i,j} (T[i, j, k] - sum_r U[i, j, r] V[j, k, r] W[k, i, r])^2
                # For fixed k:
                # A_mat3[i, j, r] = U[i, j, r] * V[j, k, r]
                # We want W[k, i, r] such that sum_r A_mat3[i, j, r] W[k, i, r] approx T[i, j, k]
                # Group by i.
                # For each i, we have a system for W[k, i, :].
                # Minimize sum_j (T[i, j, k] - sum_r U[i, j, r] V[j, k, r] W[k, i, r])^2
                # Let A_mat4[j, r] = U[i, j, r] * V[j, k, r]
                # We want W_vec[r] such that A_mat4 @ W_vec approx T[i, :, k]
                
                for k in range(p):
                    for i in range(n):
                        A_mat4 = np.zeros((m, r))
                        for j in range(m):
                            for r_idx in range(r):
                                A_mat4[j, r_idx] = U[i, j, r_idx] * V[j, k, r_idx]
                        b_vec4 = T[i, :, k]
                        try:
                            W_new, residuals, rank, sv = np.linalg.lstsq(A_mat4, b_vec4, rcond=None)
                            W[k, i, :] = W_new
                        except:
                            pass
                
                # Calculate error
                # T[i, j, k] approx sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r_idx in range(r):
                                s += U[i, j, r_idx] * V[j, k, r_idx] * W[k, i, r_idx]
                            err += (T[i, j, k] - s) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if err < 1e-6:
                    break

        if best_err < 1e-4:
            return best_U, best_V, best_W, best_err
        return None

    # Try ranks from low to high
    # Lower bound is max(n*m, m*p, p*n) ? No, rank can be lower.
    # But rank must be at least the tensor rank.
    # For 2x4x6, schoolbook is 48. Strassen-based is 42.
    # Let's try ranks 30 to 45.
    
    best_result, best_rank = schoolbook()
    
    # We will try to find a decomposition with rank < 42.
    # If we find one with rank 41, 40, etc., we convert to fractions.
    
    for r in range(35, 42):
        if time.time() > deadline:
            break
            
        res = try_rank(r, num_restarts=10, max_iters=200)
        if res:
            U, V, W, err = res
            # Convert to Fractions
            # Check if entries are close to integers or half-integers
            U_frac = [[0]*r for _ in range(n*m)]
            V_frac = [[0]*r for _ in range(m*p)]
            W_frac = [[0]*r for _ in range(p*n)]
            
            valid = True
            for i in range(n*m):
                for c in range(r):
                    val = U[i, c]
                    # Round to nearest 0.5
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) > 1e-3:
                        valid = False
                        break
                    U_frac[i][c] = Fraction(rounded).limit_denominator(100)
                if not valid: break
                
            if not valid: continue
            for j in range(m*p):
                for c in range(r):
                    val = V[j, c]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) > 1e-3:
                        valid = False
                        break
                    V_frac[j][c] = Fraction(rounded).limit_denominator(100)
                if not valid: break
                
            if not valid: continue
            for k in range(p*n):
                for c in range(r):
                    val = W[k, c]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) > 1e-3:
                        valid = False
                        break
                    W_frac[k][c] = Fraction(rounded).limit_denominator(100)
                if not valid: break
            
            if valid:
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(r):
                                s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                            if s != 1:
                                exact = False
                                break
                        if not exact: break
                    if not exact: break
                    if not exact: break
                
                if exact:
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = r
                    break # Found a better rank, stop

    return best_result

# EVOLVE-BLOCK-END
