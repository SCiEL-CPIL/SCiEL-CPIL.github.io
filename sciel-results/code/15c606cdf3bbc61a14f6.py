"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Build the multiplication tensor T with T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T[i,j,k] = 1 for all i,j,k since every A[i,j]*B[j,k] contributes to C[i,k]
    # The tensor has shape (n, m, p) = (2, 5, 5)
    
    def time_left():
        return deadline - time.time()
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly reproduces the multiplication tensor."""
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, n, m, p):
        """Faster verification using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def try_rank_hooi(r, n, m, p, max_iters=50, tol=1e-12):
        """Try to find a rank-r decomposition using HOOI-like iteration."""
        if time_left() < 5:
            return None
        
        # Initialize factors randomly
        np.random.seed(42)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time_left() < 5:
                return None
            
            # Update U: for each row i*m+j, U[i*m+j, :] = sum_{k,c} V[j*p+k,c] * W[k*n+i,c] * W[k*n+i,c]
            # Actually, the gradient of ||T - sum_c U_c ⊗ V_c ⊗ W_c||_F^2 w.r.t. U[i*m+j, c]
            # is -2 * sum_k V[j*p+k, c] * W[k*n+i, c]
            # So we project: U_new[i*m+j, c] = sum_k V[j*p+k, c] * W[k*n+i, c]
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_new[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
            
            # Similarly for V
            V_new = np.zeros_like(V)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        V_new[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
            
            # Similarly for W
            W_new = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        W_new[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
            
            # Check convergence
            delta_U = np.linalg.norm(U_new - U)
            delta_V = np.linalg.norm(V_new - V)
            delta_W = np.linalg.norm(W_new - W)
            
            U, V, W = U_new, V_new, W_new
            
            if delta_U < tol and delta_V < tol and delta_W < tol:
                break
        
        # Now we need to scale the factors so that the decomposition is exact
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # Let's check if we can find scalings alpha_c, beta_c, gamma_c such that
        # sum_c (alpha_c * U[i*m+j, c]) * (beta_c * V[j*p+k, c]) * (gamma_c * W[k*n+i, c]) = 1
        # This is equivalent to: sum_c (alpha_c * beta_c * gamma_c) * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Let s_c = alpha_c * beta_c * gamma_c, then we need sum_c s_c * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Check if the current unscaled decomposition gives the right values
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-10:
            # Convert to list of lists with exact values if possible
            U_list = [[float(x) for x in U[row, :]] for row in range(n*m)]
            V_list = [[float(x) for x in V[row, :]] for row in range(m*p)]
            W_list = [[float(x) for x in W[row, :]] for row in range(p*n)]
            return U_list, V_list, W_list
        else:
            return None
    
    def try_rank_random_search(r, n, m, p, n_trials=100):
        """Try random initializations with HOOI."""
        best_result = None
        for trial in range(n_trials):
            if time_left() < 5:
                break
            
            np.random.seed(42 + trial)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            U /= np.linalg.norm(U) + 1e-10
            V /= np.linalg.norm(V) + 1e-10
            W /= np.linalg.norm(W) + 1e-10
            
            for it in range(30):
                if time_left() < 5:
                    break
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_new[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
                
                V_new = np.zeros_like(V)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            V_new[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
                
                W_new = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            W_new[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if np.linalg.norm(U_new - U) < 1e-12:
                    break
            
            # Check if this gives an exact decomposition
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-10:
                U_list = [[float(x) for x in U[row, :]] for row in range(n*m)]
                V_list = [[float(x) for x in V[row, :]] for row in range(m*p)]
                W_list = [[float(x) for x in W[row, :]] for row in range(p*n)]
                return U_list, V_list, W_list
        
        return None
    
    def schoolbook_decomposition(n, m, p):
        """Return the schoolbook rank n*m*p decomposition."""
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
        
        return U, V, W
    
    # Try decreasing ranks from 21 upward
    # Known lower bounds suggest rank >= 21 for 2x5x5 (based on algebraic complexity)
    best_result = None
    
    for r in range(21, 51):
        if time_left() < 10:
            break
        
        # Try HOOI with random initializations
        result = try_rank_random_search(r, n, m, p, n_trials=20)
        if result is not None:
            U, V, W = result
            # Verify
            if verify_decomposition_fast(U, V, W, n, m, p):
                best_result = (r, U, V, W)
                break
        
        if time_left() < 10:
            break
    
    if best_result is not None:
        r, U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
