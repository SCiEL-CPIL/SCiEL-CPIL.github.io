import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]
N, M, P = 3, 3, 4
SCHOOLBOOK_RANK = N * M * P  # 36

def _build_target():
    """
    Build the target tensor T of shape (N*M, M*P, P*N).
    T[i*M+j, j'*P+k, k'*N+i'] = 1 iff i==i' and j==j' and k==k'
    """
    T = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    """
    Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c.
    U: (N*M, r), V: (M*P, r), W: (P*N, r)
    """
    r = U.shape[1]
    # D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
    D = np.zeros_like(T)
    for rr in range(r):
        D += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    return np.allclose(D, T, atol=tol)

def _schoolbook():
    """Standard schoolbook decomposition of rank N*M*P."""
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N*M, rank), dtype=float)
    V = np.zeros((M*P, rank), dtype=float)
    W = np.zeros((P*N, rank), dtype=float)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M + j, col] = 1
                V[j*P + k, col] = 1
                W[k*N + i, col] = 1
                col += 1
    return U, V, W

def _compute_loss(params, r):
    """
    Compute the squared error of the tensor decomposition.
    params is a flat array of length (N*M + M*P + P*N) * r.
    """
    size_U = N * M * r
    size_V = M * P * r
    size_W = P * N * r
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape(N*M, r)
    V = V_flat.reshape(M*P, r)
    W = W_flat.reshape(P*N, r)
    
    # Compute D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    D = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for rr in range(r):
        D += np.einsum('a,b,c->abc', U[:, rr], V[:, rr], W[:, rr])
    
    # Target tensor
    T = _build_target()
    
    diff = D - T
    return np.sum(diff * diff)

def _try_rank_optimization(r, max_restarts=5, deadline=None):
    """
    Try to find a rank-r decomposition using scipy.optimize.minimize.
    Returns (U, V, W) if successful, else None.
    """
    from scipy.optimize import minimize
    
    size_U = N * M * r
    size_V = M * P * r
    size_W = P * N * r
    total_size = size_U + size_V + size_W
    
    for restart in range(max_restarts):
        if deadline and time.time() > deadline - 10:
            break
        
        # Initialize with random small values
        init_vals = np.random.normal(0, 0.5, size=total_size)
        
        try:
            result = minimize(_compute_loss, init_vals, args=(r,), method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
        except Exception:
            continue
        
        if result.fun < 1e-6:
            params = result.x
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U = U_flat.reshape(N*M, r)
            V = V_flat.reshape(M*P, r)
            W = W_flat.reshape(P*N, r)
            
            # Verify
            T = _build_target()
            if _verify(U, V, W, T):
                return U, V, W
    
    return None

def construct():
    deadline = time.time() + 140  # 140s to be safe (150s kill)
    
    T = _build_target()
    
    # Best known: schoolbook rank 36
    U_best, V_best, W_best = _schoolbook()
    best_rank = SCHOOLBOOK_RANK
    
    # Try to find better decompositions by searching rank 35, 34, ... down to 20
    # The known lower bound for 3x3*3x3 is 21, so for 3x3*3x4 it's at least 21
    # Try ranks from 35 down to 20
    
    for r in range(35, 19, -1):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_optimization(r, max_restarts=3, deadline=deadline)
        if result:
            U_best, V_best, W_best = result
            best_rank = r
            # If we found a good rank, stop searching for lower ranks
            # (since lower ranks are harder to find)
            break
    
    # Convert to list-of-lists format for the return value
    U_list = [[U_best[i, j] for j in range(best_rank)] for i in range(N*M)]
    V_list = [[V_best[i, j] for j in range(best_rank)] for i in range(M*P)]
    W_list = [[W_best[i, j] for j in range(best_rank)] for i in range(P*N)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
