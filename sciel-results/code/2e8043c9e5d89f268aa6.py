"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    # Check orthogonality: for non-matching indices, sum should be 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            for kp2 in range(p):
                                if i == ip and j == jp and k == kp:
                                    continue
                                s = 0
                                for r in range(rank):
                                    s += U[i * m + j][r] * V[jp * p + k][r] * W[kp2 * n + ip][r]
                                if abs(s) > 1e-9:
                                    return False
    return True

def _is_half_integer(val):
    return abs(val * 2 - round(val * 2)) < 1e-9

def _snap_to_half_integer(val):
    return round(val * 2) / 2.0

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-10, seed=42):
    """
    Use ALS to find a rank-rank decomposition of the matrix multiplication tensor.
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Build the tensor as a 3D array: T[i,j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = np.inf
    
    for it in range(max_iters):
        # Update U: for each (i,j), U[i*m+j, :] = argmin ||T[i,j,:] - V[j*p:,:] @ W[k*n:,:]^T||
        # Actually, T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Fix V and W, solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                # For each k, T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                # Let A[k,r] = V[j*p+k,r] * W[k*n+i,r]
                # Then T[i,j,:] = A @ U[i*m+j, :]^T
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve: A @ u = T[i,j,:]
                u, residuals, rank_u, sv = np.linalg.lstsq(A, T[i, j, :], rcond=None)
                U[i * m + j, :] = u
        
        # Update V: for each (j,k), V[j*p+k, :] = argmin
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed j,k: T[:,j,k] = (U[:, j*m? no...])
                # T[i,j,k] for varying i: sum_r U[i*m+j,r] * W[k*n+i,r] * V[j*p+k,r]
                # Let B[i,r] = U[i*m+j,r] * W[k*n+i,r]
                # Then T[:,j,k] = B @ V[j*p+k, :]^T
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v, residuals, rank_v, sv = np.linalg.lstsq(B, T[:, j, k], rcond=None)
                V[j * p + k, :] = v
        
        # Update W: for each (k,i), W[k*n+i, :] = argmin
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed k,i: T[i,:,k] for varying j: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # Let C[j,r] = U[i*m+j,r] * V[j*p+k,r]
                # Then T[i,:,k] = C @ W[k*n+i, :]^T
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                w, residuals, rank_w, sv = np.linalg.lstsq(C, T[i, :, k], rcond=None)
                W[k * n + i, :] = w
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - T[i, j, k]) ** 2
                    # Also check non-target entries
        # Check non-target entries
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                for kp2 in range(p):
                                    if i == ip and j == jp and k == kp:
                                        continue
                                    s = 0.0
                                    for r in range(rank):
                                        s += U[i * m + j, r] * V[jp * p + k, r] * W[kp2 * n + ip, r]
                                    err += s ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def _snap_and_verify(U, V, W, n, m, p):
    """Snap to half-integers and verify."""
    rank = U.shape[1]
    U_snap = np.array([[_snap_to_half_integer(U[i, r]) for r in range(rank)] for i in range(U.shape[0])])
    V_snap = np.array([[_snap_to_half_integer(V[i, r]) for r in range(rank)] for i in range(V.shape[0])])
    W_snap = np.array([[_snap_to_half_integer(W[i, r]) for r in range(rank)] for i in range(W.shape[0])])
    
    # Convert to lists
    U_list = U_snap.tolist()
    V_list = V_snap.tolist()
    W_list = W_snap.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list, True
    
    return None, None, None, False

def construct():
    deadline = time.time() + 110  # 110s deadline, 10s safety margin
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    best_result = schoolbook_result
    best_rank = n * m * p  # 48
    
    # Try various ranks below schoolbook
    for rank in range(24, best_rank):  # Try ranks from 24 to 47
        if time.time() > deadline - 20:
            break
        
        # Try a few random seeds for this rank
        for seed in range(0, 5):
            if time.time() > deadline - 15:
                break
            
            try:
                U, V, W, err = _als_decompose(n, m, p, rank, max_iters=100, tol=1e-10, seed=seed)
                if err < 1e-6:
                    # Try to snap to half-integers
                    U_snap, V_snap, W_snap, valid = _snap_and_verify(U, V, W, n, m, p)
                    if valid:
                        result = {"U": U_snap, "V": V_snap, "W": W_snap}
                        best_result = result
                        best_rank = rank
                        break
            except Exception:
                continue
        
        if best_rank <= rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
