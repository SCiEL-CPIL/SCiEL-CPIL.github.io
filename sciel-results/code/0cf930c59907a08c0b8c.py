"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import numpy as np
import time

_START = time.time()
_LIMIT = 130.0          # hard safety margin before the 150 s wall-clock kill
DIMENSIONS = [3, 4, 7]


def _matmul_vec(U, V, W, n, m, p, r):
    """Bilinear output for coefficient 1: out[i,k] = sum_j A[i,j]B[j,k] where
       A[i,j] = U[i*m+j,0], B[j,k] = V[j*p+k,0], C[i,k] = W[k*n+i,0]."""
    A = U.reshape(n, m, r)[:, :, 0]
    B = V.reshape(m, p, r)[:, :, 0]
    C = W.reshape(p, n, r)[:, :, 0]
    return A @ B - C


def _check(U, V, W, n, m, p, r, tol=1e-6):
    ok = True
    for c in range(1, r):
        if np.linalg.norm(_matmul_vec(U[:, c], V[:, c], W[:, c], n, m, p)) > tol:
            ok = False
            break
    return ok


def construct():
    n, m, p = DIMENSIONS
    Nm, Mp, Pn = n * m, m * p, p * n

    # --- 1. Explicit schoolbook decomposition (rank n*m*p), always valid ---
    R0 = n * m * p
    U0 = np.zeros((Nm, R0)); V0 = np.zeros((Mp, R0)); W0 = np.zeros((Pn, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    def _pack(arr):
        return [[float(x) for x in row] for row in np.asarray(arr)]

    # best result found so far (initially schoolbook)
    best = {"U": _pack(U0), "V": _pack(V0), "W": _pack(W0)}

    # --- 2. Search for a lower-rank exact decomposition ---
    # We know rank >= max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # But also rank >= n+m+p-2 = 12 (for general matmul)
    # The actual lower bound is max(n*p, m*p, n*m) for the standard tensor rank.
    # For 3x4x7: lower bound is 28.
    
    # Actually, the tensor rank of matmul is at least max(n*p, m*p, n*m)
    # = max(21, 28, 12) = 28.
    
    # Let's try to find rank 28 decomposition.
    # The strategy: try random initializations and use ALS to minimize the residual.
    # If we can get residual to 0 (within tolerance), we have an exact decomposition.
    
    # Build the multiplication tensor T as a 3-way tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0

    # Reshape T for ALS
    # We want to factor T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^n, v_r is in R^m, w_r is in R^p
    
    def _als_step(U, V, W, n, m, p, r, T, max_iter=100, tol=1e-10):
        """ALS for 3-way tensor decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() > _LIMIT - 30.0:
                break
            
            # Update U
            for rr in range(r):
                # Project T onto V[:, rr] ⊗ W[:, rr]
                # T_mat[i, j, k] -> for fixed rr, compute (T * (V_rr ⊗ W_rr)) over j,k
                # T_contracted[i] = sum_{j,k} T[i,j,k] * V[j,rr] * W[k,rr]
                T_contracted = np.einsum('ijk,jr,kr->ir', T, V, W)
                # U[i, rr] = T_contracted[i, rr] / (V[j,rr]^2 * W[k,rr]^2)
                # Actually: U[i, rr] = sum_{j,k} T[i,j,k] V[j,rr] W[k,rr] / (||V[:,rr]||^2 ||W[:,rr]||^2)
                denom = np.sum(V[:, rr] ** 2) * np.sum(W[:, rr] ** 2)
                if denom > 1e-15:
                    U[:, rr] = T_contracted[:, rr] / denom
            
            # Update V
            for rr in range(r):
                T_contracted = np.einsum('ijk,ir,kr->jr', T, U, W)
                denom = np.sum(U[:, rr] ** 2) * np.sum(W[:, rr] ** 2)
                if denom > 1e-15:
                    V[:, rr] = T_contracted[:, rr] / denom
            
            # Update W
            for rr in range(r):
                T_contracted = np.einsum('ijk,ir,jr->kr', T, U, V)
                denom = np.sum(U[:, rr] ** 2) * np.sum(V[:, rr] ** 2)
                if denom > 1e-15:
                    W[:, rr] = T_contracted[:, rr] / denom
            
            # Check convergence
            if it > 0 and it % 10 == 0:
                residual = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', U, V, W))
                if residual < tol:
                    break
        
        return U, V, W
    
    def _check_exact(U, V, W, n, m, p, r, tol=1e-6):
        """Check if U,V,W exactly decompose the matmul tensor."""
        residual = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', U, V, W))
        return residual < tol
    
    best_rank = R0
    best_tuple = (U0, V0, W0)
    
    def _improve(U, V, W):
        nonlocal best_rank, best_tuple, best
        if _check(U, V, W, n, m, p, U.shape[1], tol=1e-6):
            best_rank = U.shape[1]
            best_tuple = (U, V, W)
            best = {"U": _pack(U), "V": _pack(V), "W": _pack(W)}
    
    # Try ranks from 28 upward
    for r in range(28, min(R0, 40)):
        if time.time() > _LIMIT - 30.0:
            break
        
        found = False
        n_restarts = 5 if r <= 30 else 3
        
        for restart in range(n_restarts):
            if time.time() > _LIMIT - 20.0:
                break
            
            seed = 10000 + r * 100 + restart
            rs = np.random.RandomState(seed)
            
            # Initialize with random vectors
            U = rs.standard_normal((n, r))
            V = rs.standard_normal((m, r))
            W = rs.standard_normal((p, r))
            
            # Run ALS
            U, V, W = _als_step(U, V, W, n, m, p, r, T, max_iter=200, tol=1e-12)
            
            # Check if exact
            if _check_exact(U, V, W, n, m, p, r, tol=1e-8):
                # Convert to the required format
                U_full = np.hstack([U0[:, :1], U.T])
                V_full = np.hstack([V0[:, :1], V.T])
                W_full = np.hstack([W0[:, :1], W.T])
                
                if _check(U_full, V_full, W_full, n, m, p, r + 1, tol=1e-6):
                    best_rank = r + 1
                    best = {"U": _pack(U_full), "V": _pack(V_full), "W": _pack(W_full)}
                    found = True
                    break
        
        if found:
            break  # Found exact at this rank, stop
    
    return best
