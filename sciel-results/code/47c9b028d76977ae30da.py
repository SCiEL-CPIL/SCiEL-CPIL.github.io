"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    deadline = time.time() + 140
    
    best_rank = n * m * p
    best_result = None
    
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1) > 1e-8:
                        return False
        return True
    
    def try_rank(rank, max_restarts=10):
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(random.randint(0, 2**31))
            
            # Initialize: use schoolbook projected to rank
            # Create random projection matrices
            P_u = np.random.randn(nm, rank)
            P_v = np.random.randn(mp, rank)
            P_w = np.random.randn(pn, rank)
            
            # Normalize
            P_u = P_u / np.linalg.norm(P_u, axis=0, keepdims=True)
            P_v = P_v / np.linalg.norm(P_v, axis=0, keepdims=True)
            P_w = P_w / np.linalg.norm(P_w, axis=0, keepdims=True)
            
            U = P_u
            V = P_v
            W = P_w
            
            # Alternating minimization
            for it in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W given U, V
                A_w = np.zeros((n, p, rank))
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            A_w[i, k, :] += U[i*m+j, :] * V[j*p+k, :]
                
                b_w = np.ones(n * p)
                A_mat = np.zeros((n * p, rank))
                for i in range(n):
                    for k in range(p):
                        A_mat[k*n+i, :] = A_w[i, k, :]
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_mat, b_w, rcond=None)
                except:
                    return None
                
                # Solve for U given V, W
                A_u = np.zeros((n, m, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A_u[i, j, :] += V[j*p+k, :] * W[k*n+i, :]
                
                b_u = np.ones(n * m)
                A_mat_u = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        A_mat_u[i*m+j, :] = A_u[i, j, :]
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_mat_u, b_u, rcond=None)
                except:
                    return None
                
                # Solve for V given U, W
                A_v = np.zeros((m, p, rank))
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            A_v[j, k, :] += U[i*m+j, :] * W[k*n+i, :]
                
                b_v = np.ones(m * p)
                A_mat_v = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        A_mat_v[j*p+k, :] = A_v[j, k, :]
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_mat_v, b_v, rcond=None)
                except:
                    return None
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if verify(U, V, W):
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if verify(U_r, V_r, W_r):
                        return (U_r, V_r, W_r)
                    # Check if already half-integer
                    all_half = True
                    for mat in [U, V, W]:
                        for row in mat:
                            for val in row:
                                if abs(val * 2 - round(val * 2)) > 1e-6:
                                    all_half = False
                                    break
                            if not all_half:
                                break
                        if not all_half:
                            break
                    if all_half:
                        return (np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2)
                    # Return as-is if close enough
                    if verify(U, V, W):
                        return (U, V, W)
            
            # Final check
            if verify(U, V, W):
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
                all_half = True
                for mat in [U, V, W]:
                    for row in mat:
                        for val in row:
                            if abs(val * 2 - round(val * 2)) > 1e-6:
                                all_half = False
                                break
                            if not all_half:
                                break
                        if not all_half:
                            break
                if all_half:
                    return (np.round(U*2)/2, np.round(V*2)/2, np.round(W*2)/2)
                return (U, V, W)
        
        return None
    
    # Try ranks from 30 upward
    for rank in range(30, n*m*p + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_restarts=5)
        if result is not None:
            U, V, W = result
            # Final verification
            if verify(U, V, W):
                best_rank = rank
                best_result = {
                    "U": U.tolist() if isinstance(U, np.ndarray) else U,
                    "V": V.tolist() if isinstance(V, np.ndarray) else V,
                    "W": W.tolist() if isinstance(W, np.ndarray) else W
                }
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
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

