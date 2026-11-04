"""Matrix multiplication 2x5x5 rank minimization via randomized ALS search."""
DIMENSIONS = [2, 5, 5]

import numpy as np
import time

def _build_tensor():
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    where T[i*m+j, j*p+k, k*n+i] = 1.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _als_decompose(T, rank, max_iter=200, tol=1e-9):
    """
    Try to find a decomposition of rank `rank` using Alternating Least Squares (ALS).
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    for it in range(max_iter):
        # Update U
        for r in range(rank):
            num = np.einsum('abc,b,c->a', T, V[:, r], W[:, r])
            den = np.sum(V[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-12:
                U[:, r] = num / den
        
        # Update V
        for r in range(rank):
            num = np.einsum('abc,a,c->b', T, U[:, r], W[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(W[:, r] ** 2)
            if den > 1e-12:
                V[:, r] = num / den
        
        # Update W
        for r in range(rank):
            num = np.einsum('abc,a,b->c', T, U[:, r], V[:, r])
            den = np.sum(U[:, r] ** 2) * np.sum(V[:, r] ** 2)
            if den > 1e-12:
                W[:, r] = num / den
        
        # Compute error
        recon = np.zeros_like(T)
        for r in range(rank):
            recon += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        err = np.linalg.norm(recon - T)
        if err < tol:
            return U, V, W, True
    
    return U, V, W, False

def _snap_and_verify(U, V, W, T, tol=1e-9):
    """
    Snap the matrices to half-integers and verify the decomposition.
    """
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    
    recon = np.zeros_like(T)
    for r in range(U.shape[1]):
        recon += U_s[:, r][:, None, None] * V_s[:, r][None, :, None] * W_s[:, r][None, None, :]
    
    if np.allclose(recon, T, atol=tol):
        return U_s, V_s, W_s, True
    return None, None, None, False

def _schoolbook_decomposition():
    """Return the standard schoolbook decomposition of rank 50."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with shapes (10, rank), (25, rank), (10, rank).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    T = _build_tensor()
    
    # Try ranks from 45 to 50
    # Known lower bound is 25, but 45 is a strong candidate for minimum
    for rank in range(45, 51):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Multiple random restarts to escape local minima
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            try:
                U, V, W, success = _als_decompose(T, rank, max_iter=150, tol=1e-9)
                if success:
                    U_s, V_s, W_s, valid = _snap_and_verify(U, V, W, T)
                    if valid:
                        return {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }
            except Exception:
                pass
        
        # If rank not found, continue to next rank
    
    # Fallback to schoolbook decomposition (rank 50)
    U, V, W = _schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
