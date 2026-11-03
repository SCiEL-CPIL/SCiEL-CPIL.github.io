"""Rewritten program for matrix multiplication 3x3x7 rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_fast(U, V, W, n=None, m=None, p=None, tol=1e-9):
    """Faster verification using vectorized operations."""
    if n is None or m is None or p is None:
        n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        u_3d = u_col.reshape(n, m)
        v_3d = v_col.reshape(m, p)
        w_3d = w_col.reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_3d[i, j] * v_3d[j, k] * w_3d[k, i]
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol


def _alternating_minimize(n, m, p, r, max_iters=200, tol=1e-12, deadline=None):
    """
    Alternating minimization for tensor decomposition.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all valid (i,j,k)
    """
    np.random.seed(42)
    
    # Random initialization
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each row a = i*m+j, solve U[a,:] such that
        # sum_c U[a,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is a linear system: U[a,:] @ M_a = ones(p)
        # where M_a[k,c] = V[j*p+k,c] * W[k*n+i,c]
        
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                a = i * m + j
                # Build M_a: shape (p, r)
                M_a = np.zeros((p, r))
                for k in range(p):
                    M_a[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve new_U[a,:] @ M_a = ones(p)
                # i.e., M_a.T @ new_U[a,:] = ones(p)
                try:
                    new_U[a, :] = np.linalg.lstsq(M_a.T, np.ones(p), rcond=None)[0]
                except:
                    new_U[a, :] = U[a, :]
        U = new_U
        
        # Update V: for each row b = j*p+k, solve V[b,:] such that
        # sum_c U[i*m+j,c] * V[b,c] * W[k*n+i,c] = 1 for all i
        # M_b[i,c] = U[i*m+j,c] * W[k*n+i,c]
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M_b = np.zeros((n, r))
                for i in range(n):
                    M_b[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    new_V[b, :] = np.linalg.lstsq(M_b.T, np.ones(n), rcond=None)[0]
                except:
                    new_V[b, :] = V[b, :]
        V = new_V
        
        # Update W: for each row c = k*n+i, solve W[c,:] such that
        # sum_c' U[i*m+j,c'] * V[j*p+k,c'] * W[c,:] = 1 for all j
        # M_c[j,c'] = U[i*m+j,c'] * V[j*p+k,c']
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M_c = np.zeros((m, r))
                for j in range(m):
                    M_c[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    new_W[c, :] = np.linalg.lstsq(M_c.T, np.ones(m), rcond=None)[0]
                except:
                    new_W[c, :] = W[c, :]
        W = new_W
        
        # Compute error
        T = np.zeros((n, m, p))
        for r_idx in range(r):
            u_3d = U[:, r_idx].reshape(n, m)
            v_3d = V[:, r_idx].reshape(m, p)
            w_3d = W[:, r_idx].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_3d[i, j] * v_3d[j, k] * w_3d[k, i]
        
        err = np.max(np.abs(T - 1.0))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err


def _round_to_half_integer(X, tol=1e-6):
    """Round entries to nearest half-integer if close enough."""
    X_round = np.round(X * 2) / 2
    if np.max(np.abs(X - X_round)) < tol:
        return X_round
    return X


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication.
    Returns {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as the baseline
    best_result = _schoolbook()
    best_rank = n * m * p  # 63
    
    # Verify the schoolbook result
    if _verify_fast(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        pass
    
    # Try alternating minimization with various ranks
    # Lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Try ranks from 21 up to 62
    
    candidate_ranks = list(range(21, 63))
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for seed in range(8):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed * 1000 + r + 7)
            U, V, W, err = _alternating_minimize(n, m, p, r, max_iters=150, tol=1e-10, deadline=deadline - 5)
            
            if err < 1e-8:
                # Found a valid (or near-valid) decomposition
                # Try to round to half-integers
                U_round = _round_to_half_integer(U)
                V_round = _round_to_half_integer(V)
                W_round = _round_to_half_integer(W)
                
                if _verify_fast(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p, tol=1e-6):
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    break
                else:
                    # Use the unrounded solution
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    # Return the best result found
    return best_result

# EVOLVE-BLOCK-END
