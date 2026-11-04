import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
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

    start_time = time.time()
    deadline = start_time + 115.0

    # Try to find a lower rank decomposition
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40, 44]
    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 10:
            break
        try:
            result = search_rank(n, m, p, target_rank, deadline)
            if result is not None:
                return result
        except Exception:
            pass

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def search_rank(n, m, p, rank, deadline):
    """
    Search for a rank-r decomposition using Alternating Least Squares (ALS).
    """
    best_result = None
    best_error = float('inf')
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random integers
        U = np.random.randint(-1, 2, size=(n * m, rank)).astype(float)
        V = np.random.randint(-1, 2, size=(m * p, rank)).astype(float)
        W = np.random.randint(-1, 2, size=(p * n, rank)).astype(float)
        
        # Run ALS
        U, V, W = als_solve(U, V, W, n, m, p, rank, deadline)
        
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        error = compute_error(U_snap, V_snap, W_snap, n, m, p)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
            
        if error < 1e-10:
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
            
    if best_error < 1e-10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    return None

def als_solve(U, V, W, n, m, p, rank, deadline, max_iters=50):
    """
    Alternating Least Squares for tensor decomposition.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Target T[i,j,k] = 1.
    """
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        # Update U: Fix V, W
        # For each i, j, solve A U_ij = b
        # A[k, r] = V3[j, k, r] * W3[k, i, r]
        # b[k] = 1
        for i in range(n):
            for j in range(m):
                # Construct A
                # V3[j] is (p, rank)
                # W3[:, i] is (p, rank)
                # A = V3[j] * W3[:, i] (element-wise product, shape p x rank)
                A = V3[j] * W3[:, i]
                b = np.ones(p)
                # Solve A x = b
                # Use least squares
                try:
                    U3[i, j] = np.linalg.lstsq(A, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
                    
        # Update V: Fix U, W
        # For each j, k, solve B V_jk = b
        # B[i, r] = U3[i, j, r] * W3[k, i, r]
        # b[i] = 1
        for j in range(m):
            for k in range(p):
                # Construct B
                # U3[:, j] is (n, rank)
                # W3[k, :] is (n, rank)
                # B = U3[:, j] * W3[k, :] (element-wise product, shape n x rank)
                B = U3[:, j] * W3[k, :]
                b = np.ones(n)
                try:
                    V3[j, k] = np.linalg.lstsq(B, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
                    
        # Update W: Fix U, V
        # For each k, i, solve C W_ki = b
        # C[j, r] = U3[i, j, r] * V3[j, k, r]
        # b[j] = 1
        for k in range(p):
            for i in range(n):
                # Construct C
                # U3[i, :] is (m, rank)
                # V3[:, k] is (m, rank)
                # C = U3[i, :] * V3[:, k] (element-wise product, shape m x rank)
                C = U3[i, :] * V3[:, k]
                b = np.ones(m)
                try:
                    W3[k, i] = np.linalg.lstsq(C, b, rcond=None)[0]
                except np.linalg.LinAlgError:
                    pass
                    
    U = U3.reshape(n * m, rank)
    V = V3.reshape(m * p, rank)
    W = W3.reshape(p * n, rank)
    
    return U, V, W

def compute_error(U, V, W, n, m, p):
    """
    Compute the Frobenius norm squared of the difference between the decomposed tensor
    and the target tensor (all ones).
    """
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    error = np.sum((T - 1.0)**2)
    return error
