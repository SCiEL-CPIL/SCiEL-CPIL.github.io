"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def _compute_T_comp(U, V, W, n, m, p, rank):
    """Compute T_comp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    T_comp = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            u_i = u_col[i*m:(i+1)*m]
            for j in range(m):
                v_j = v_col[j*p:(j+1)*p]
                for k in range(p):
                    T_comp[i, j, k] += u_i[j] * v_j[k] * w_col[k*n + i]
    return T_comp

def _verify(U, V, W, n, m, p, rank, tol=1e-8):
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T[i, j, i] = 1.0
    T_comp = _compute_T_comp(U, V, W, n, m, p, rank)
    return np.max(np.abs(T_comp - T)) < tol, T_comp

def _kronecker_decomposition(n, m, p):
    """Construct rank-(n*p) Kronecker decomposition for T[i,j,k]=delta_{i,k}."""
    rank = n * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for k in range(p):
            r = i * p + k
            for j in range(m):
                U[i * m + j, r] = 1.0
            for jj in range(m):
                V[jj * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Target tensor: T[i,j,k] = 1 if i == k, else 0
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T[i, j, i] = 1.0
    
    # Kronecker decomposition (rank 10)
    U_kron, V_kron, W_kron, rank_kron = _kronecker_decomposition(n, m, p)
    
    # Verify Kronecker decomposition
    ok, _ = _verify(U_kron, V_kron, W_kron, n, m, p, rank_kron)
    if not ok:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U = U_kron
    best_V = V_kron
    best_W = W_kron
    best_rank = rank_kron
    
    # Try to find lower rank decompositions
    # Lower bound: rank >= max(n, m, p) = 5, and rank >= n*p = 10 for this structure
    # Actually for T[i,j,k]=delta_{i,k}, the rank is at least n*p = 10
    # Let's try ranks from 9 down to 5
    
    for target_rank in range(rank_kron - 1, 4, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_tries = 200
        
        for attempt in range(max_tries):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-1, 2, size=(n * m, target_rank)).astype(float)
            V = np.random.randint(-1, 2, size=(m * p, target_rank)).astype(float)
            W = np.random.randint(-1, 2, size=(p * n, target_rank)).astype(float)
            
            # Check initial
            ok, _ = _verify(U, V, W, n, m, p, target_rank)
            if ok:
                found = True
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
            
            # Newton-like iterations: fix two matrices, solve for the third
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                
                # Solve for W given U, V
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed i,k: T[i,j,k] = sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
                # Let A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then T[i,:,k] = A @ W[k*n+i, :]
                # W[k*n+i, :] = A^{-1} @ T[i,:,k]
                
                W_new = W.copy()
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        try:
                            W_new[k * n + i, :] = np.linalg.solve(A, T[i, :, k])
                        except np.linalg.LinAlgError:
                            pass
                
                # Check
                ok, _ = _verify(U, V, W_new, n, m, p, target_rank)
                if ok:
                    found = True
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W_new
                    break
                
                # Solve for U given V, W
                U_new = U.copy()
                for j in range(m):
                    for r_row in range(target_rank):
                        pass
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed j: T[:,j,:] = sum_r U[:,j,r] * (V[j,*,r] * W[*,r])
                # Let B[i, r] = V[j*p+*, r] * W[*, r] reshaped
                # Actually: T[i,j,k] = sum_r U[i*m+j, r] * C_j[i, k, r] where C_j[i,k,r] = V[j*p+k,r]*W[k*n+i,r]
                # T[:,j,:] = sum_r U[:,j,r] * C_j[:,:,r]
                # This is a linear system in U[:,j,:]
                
                for j in range(m):
                    # Build system: for each (i,k), T[i,j,k] = sum_r U[i*m+j, r] * C_j[i,k,r]
                    # A[j] is (n*p) x target_rank
                    A_j = np.zeros((n * p, target_rank))
                    for i in range(n):
                        for k in range(p):
                            for r in range(target_rank):
                                A_j[i * p + k, r] = V[j * p + k, r] * W[k * n + i, r]
                    b_j = T[:, j, :].flatten()
                    try:
                        U_new[j::m, :] = np.linalg.solve(A_j, b_j)
                    except np.linalg.LinAlgError:
                        pass
                
                ok, _ = _verify(U_new, V, W_new, n, m, p, target_rank)
                if ok:
                    found = True
                    best_rank = target_rank
                    best_U = U_new
                    best_V = V
                    best_W = W_new
                    break
                
                # Solve for V given U, W
                V_new = V.copy()
                for j in range(m):
                    A_j = np.zeros((n * p, target_rank))
                    for i in range(n):
                        for k in range(p):
                            for r in range(target_rank):
                                A_j[i * p + k, r] = U[i * m + j, r] * W[k * n + i, r]
                    b_j = T[:, j, :].flatten()
                    try:
                        V_new[j * p:(j+1) * p, :] = np.linalg.solve(A_j, b_j)
                    except np.linalg.LinAlgError:
                        pass
                
                ok, _ = _verify(U_new, V_new, W_new, n, m, p, target_rank)
                if ok:
                    found = True
                    best_rank = target_rank
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    break
                
                U, V, W = U_new, V_new, W_new
        
        if found:
            break
    
    # Try to round to integer/half-integer
    for scale, name in [(1, 'int'), (2, 'half')]:
        U_r = np.round(best_U * scale) / scale
        V_r = np.round(best_V * scale) / scale
        W_r = np.round(best_W * scale) / scale
        ok, _ = _verify(U_r, V_r, W_r, n, m, p, best_rank)
        if ok:
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
