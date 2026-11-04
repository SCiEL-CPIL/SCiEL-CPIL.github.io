"""
Matrix multiplication 3x4 by 4x8: rank-minimal exact tensor decomposition.
Uses a randomized search with local refinement to find a low-rank decomposition.
The schoolbook rank is 3*4*8 = 96. We try to find a decomposition with rank < 96.
"""

import numpy as np
import time
import itertools
from copy import deepcopy

DIMENSIONS = [3, 4, 8]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k].
    We need T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    and W[k*n+i, c] is the convention.
    """
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using numpy."""
    rank = U.shape[1]
    # For each c, the contribution to T[i,j,k] is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank(n, m, p, rank, max_iter=200, seed=42):
    """Try to find a rank decomposition. Returns (U,V,W) or None."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for it in range(max_iter):
        # Initialize randomly
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize
        for c in range(rank):
            norm = np.linalg.norm(U[:, c]) * np.linalg.norm(V[:, c]) * np.linalg.norm(W[:, c])
            if norm > 1e-12:
                U[:, c] /= norm**(1/3)
                V[:, c] /= norm**(1/3)
                W[:, c] /= norm**(1/3)
        
        # Alternating least squares
        for inner in range(30):
            # Update U
            for c in range(rank):
                # For each (i,j), we need U[i*m+j, c] = sum_k T[i,j,k] * V[j*p+k,c] * W[k*n+i,c] / sum_k V[j*p+k,c]^2 * W[k*n+i,c]^2
                # Actually, the objective is sum_{i,j,k} (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
                # For fixed V,W, U[i*m+j,c] = sum_k T[i,j,k]*V[j*p+k,c]*W[k*n+i,c] / sum_k V[j*p+k,c]^2*W[k*n+i,c]^2
                for i in range(n):
                    for j in range(m):
                        num = 0.0
                        den = 0.0
                        for k in range(p):
                            num += T[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
                            den += V[j*p+k, c]**2 * W[k*n+i, c]**2
                        if den > 1e-12:
                            U[i*m+j, c] = num / den
                        else:
                            U[i*m+j, c] = 0.0
            
            # Update V
            for c in range(rank):
                for j in range(m):
                    for k in range(p):
                        num = 0.0
                        den = 0.0
                        for i in range(n):
                            num += T[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
                            den += U[i*m+j, c]**2 * W[k*n+i, c]**2
                        if den > 1e-12:
                            V[j*p+k, c] = num / den
                        else:
                            V[j*p+k, c] = 0.0
            
            # Update W
            for c in range(rank):
                for k in range(p):
                    for i in range(n):
                        num = 0.0
                        den = 0.0
                        for j in range(m):
                            num += T[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
                            den += U[i*m+j, c]**2 * V[j*p+k, c]**2
                        if den > 1e-12:
                            W[k*n+i, c] = num / den
                        else:
                            W[k*n+i, c] = 0.0
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - T[i,j,k])**2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-6:
            break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def _try_binary(n, m, p, rank, max_iter=100, seed=42):
    """Try to find a rank decomposition with entries in {-1, 0, 1} or half-integers."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    for it in range(max_iter):
        U = rng.choice([-1, 0, 1], size=(n*m, rank))
        V = rng.choice([-1, 0, 1], size=(m*p, rank))
        W = rng.choice([-1, 0, 1], size=(p*n, rank))
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0)**2
        
        if err < 1e-6:
            return U.astype(float), V.astype(float), W.astype(float)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank
    # Theoretical lower bounds: rank >= max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Also rank >= n*p = 24 (from output dimension)
    # Let's try ranks from 32 upward
    
    for rank in range(32, best_rank):
        if time.time() > deadline:
            break
        
        # Try randomized ALS
        result = _try_rank(n, m, p, rank, max_iter=50, seed=rank)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        
        # Try binary
        if time.time() > deadline:
            break
        result = _try_binary(n, m, p, rank, max_iter=200, seed=rank+1000)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    # Convert to lists of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
