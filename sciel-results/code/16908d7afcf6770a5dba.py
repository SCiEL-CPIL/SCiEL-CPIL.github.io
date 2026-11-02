"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def check_decomposition(U, V, W, tol=1e-6):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half_integers(U, V, W):
        U = np.round(np.array(U, dtype=float) * 2) / 2
        V = np.round(np.array(V, dtype=float) * 2) / 2
        W = np.round(np.array(W, dtype=float) * 2) / 2
        return U.tolist(), V.tolist(), W.tolist()
    
    # Precompute indices
    u_idx = np.array([i*m+j for i in range(n) for j in range(m) for k in range(p)])
    v_idx = np.array([j*p+k for i in range(n) for j in range(m) for k in range(p)])
    w_idx = np.array([k*n+i for i in range(n) for j in range(m) for k in range(p)])
    num_eq = 63
    
    def coordinate_descent(U, V, W, max_iter=2000):
        """Coordinate descent: fix two matrices, solve for the third."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for W given U, V
            # For each (i,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # This is: for each (i,k), we need W[k*n+i, :] to satisfy
            # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then we need A_ik · W[k*n+i, :] = 1
            # This is a single equation in r unknowns - underdetermined
            # Use least squares or just set W[k*n+i, r] = A_ik[r] / (A_ik · A_ik)
            
            # Actually, for each (i,k), we have one equation: A_ik · w_ik = 1
            # where w_ik = W[k*n+i, :]
            # Solution: w_ik = A_ik / (A_ik · A_ik) if A_ik is not zero
            
            for i in range(n):
                for k in range(p):
                    A_ik = np.zeros(r)
                    for j in range(m):
                        A_ik += U[i*m+j, :] * V[j*p+k, :]
                    norm = np.dot(A_ik, A_ik)
                    if norm > 1e-12:
                        W[k*n+i, :] = A_ik / norm
                    else:
                        W[k*n+i, :] = 1.0 / r
            
            # Similarly for U and V
            # For U: for each (i,j), sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let B_ij[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            # Then B_ij · U[i*m+j, :] = 1
            for i in range(n):
                for j in range(m):
                    B_ij = np.zeros(r)
                    for k in range(p):
                        B_ij += V[j*p+k, :] * W[k*n+i, :]
                    norm = np.dot(B_ij, B_ij)
                    if norm > 1e-12:
                        U[i*m+j, :] = B_ij / norm
                    else:
                        U[i*m+j, :] = 1.0 / r
            
            # For V: for each (j,k), sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let C_jk[r] = sum_i U[i*m+j, r] * W[k*n+i, r]
            # Then C_jk · V[j*p+k, :] = 1
            for j in range(m):
                for k in range(p):
                    C_jk = np.zeros(r)
                    for i in range(n):
                        C_jk += U[i*m+j, :] * W[k*n+i, :]
                    norm = np.dot(C_jk, C_jk)
                    if norm > 1e-12:
                        V[j*p+k, :] = C_jk / norm
                    else:
                        V[j*p+k, :] = 1.0 / r
            
            # Check convergence
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                break
        
        return U, V, W, max_err
    
    def try_rank(r, num_restarts=5):
        U_dim = n * m
        V_dim = m * p
        W_dim = p * n
        
        best_err = float('inf')
        best_UVW = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(U_dim, r) * 0.5
            V = np.random.randn(V_dim, r) * 0.5
            W = np.random.randn(W_dim, r) * 0.5
            
            U, V, W, err = coordinate_descent(U, V, W, max_iter=500)
            
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-8:
                break
        
        if best_UVW is not None and best_err < 1e-6:
            U, V, W = best_UVW
            U_s, V_s, W_s = snap_to_half_integers(U, V, W)
            if check_decomposition(U_s, V_s, W_s, tol=1e-4):
                return U_s, V_s, W_s
            if check_decomposition(U.tolist(), V.tolist(), W.tolist(), tol=1e-4):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 21 upward
    for r in range(21, min(best_rank, 40)):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            if check_decomposition(U_r, V_r, W_r, tol=1e-4):
                best = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
