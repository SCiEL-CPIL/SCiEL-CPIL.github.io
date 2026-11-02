import time
import numpy as np
from scipy.linalg import lstsq
from itertools import product
import random

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 140.0  # safety margin before 150s kill

def _build_tensor(n, m, p):
    """
    Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1
    All other entries are 0.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _tensor_error_fast(U, V, W, T):
    """
    Compute ||sum_r U[:,r] outer V[:,r] outer W[:,r] - T||_F^2
    using einsum for efficiency.
    """
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = reconstructed - T
    return np.sum(diff * diff)

def _schoolbook_solution(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _structured_init_mm(n, m, p, rank, seed):
    """
    Create structured initializations exploiting matrix multiplication structure.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Fill with random values, then normalize columns
    U[:] = rng.randn(n*m, rank)
    V[:] = rng.randn(m*p, rank)
    W[:] = rng.randn(p*n, rank)
    
    # Normalize columns to unit norm
    for mat in [U, V, W]:
        norms = np.linalg.norm(mat, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        mat /= norms
    
    return U, V, W

def _als_step_fast(U, V, W, T, n, m, p):
    """
    Perform one ALS iteration with efficient matrix operations.
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    """
    rank = U.shape[1]
    
    # Update U: T.reshape(n*m, m*p*p*n) = U @ A_u
    # A_u[r, j*p*n + k] = V[j,r] * W[k,r]
    A_u = np.zeros((rank, m*p*p*n), dtype=np.float64)
    for r in range(rank):
        A_u[r, :] = (V[:, r] * W[:, r]).reshape(-1)
    
    T_flat_u = T.reshape(n*m, m*p*p*n)
    U_new, _, _, _ = lstsq(A_u, T_flat_u.T, cond=None)
    U = U_new.T  # (n*m, rank)
    
    # Update V: T.transpose(1,0,2).reshape(m*p, n*m*p*n) = V @ A_v
    # A_v[r, i*p*n + k] = U[i,r] * W[k,r]
    A_v = np.zeros((rank, n*m*p*n), dtype=np.float64)
    for r in range(rank):
        A_v[r, :] = (U[:, r] * W[:, r]).reshape(-1)
    
    T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    V_new, _, _, _ = lstsq(A_v, T_flat_v.T, cond=None)
    V = V_new.T  # (m*p, rank)
    
    # Update W: T.transpose(2,0,1).reshape(p*n, n*m*m*p) = W @ A_w
    # A_w[r, i*m*p + j] = U[i,r] * V[j,r]
    A_w = np.zeros((rank, n*m*m*p), dtype=np.float64)
    for r in range(rank):
        A_w[r, :] = (U[:, r] * V[:, r]).reshape(-1)
    
    T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
    W_new, _, _, _ = lstsq(A_w, T_flat_w.T, cond=None)
    W = W_new.T  # (p*n, rank)
    
    return U, V, W

def _try_rank_als(n, m, p, rank, T, seed=0, time_limit=5.0):
    """
    Attempt to find a decomposition of the given rank using ALS.
    Returns (U, V, W, error) or None if time limit exceeded without convergence.
    """
    start_time = time.time()
    
    # Multiple initializations
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for init_seed in range(seed, seed + 3):
        if time.time() - start_time > time_limit:
            break
            
        U, V, W = _structured_init_mm(n, m, p, rank, init_seed)
        
        cur_err = _tensor_error_fast(U, V, W, T)
        
        # ALS iterations
        max_iters = 2000
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = _als_step_fast(U, V, W, T, n, m, p)
            
            cur_err = _tensor_error_fast(U, V, W, T)
            
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-12:
                break
    
    if best_U is None:
        return None
    
    return best_U, best_V, best_W, best_err

def _polish_solution(U, V, W, T, n, m, p, rank, time_limit=3.0):
    """
    Polish a near-converged solution with more ALS iterations and tighter tolerance.
    """
    start_time = time.time()
    best_err = _tensor_error_fast(U, V, W, T)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    max_iters = 1000
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        U, V, W = _als_step_fast(U, V, W, T, n, m, p)
        cur_err = _tensor_error_fast(U, V, W, T)
        
        if cur_err < best_err:
            best_err = cur_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-14:
            break
    
    return best_U, best_V, best_W, best_err

def _verify_exact(U, V, W, T, n, m, p, tol=1e-10):
    """
    Verify that the decomposition is exact (or nearly exact).
    """
    err = _tensor_error_fast(U, V, W, T)
    return err < tol

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_solution(n, m, p)
    best_err = 0.0
    T = _build_tensor(n, m, p)
    
    # Try to reduce rank from n*m*p - 1 downwards
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25.
    # But known results might be lower.
    # Let's try ranks: 49, 48, 47, ..., 25.
    
    # To save time, we'll try a few promising ones with multiple seeds.
    # Ranks to try: 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25
    
    # We'll be more aggressive: try all ranks from 49 down to 25, but with limited time per rank.
    
    # First, let's see if we can find any rank < 50.
    # If not, return schoolbook.
    
    found_rank = n * m * p
    found_U, found_V, found_W = U, V, W
    
    # Try ranks in decreasing order
    for rank in range(n*m*p - 1, 24, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        # Try multiple seeds for this rank
        success = False
        for seed in range(3):
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
                
            time_limit = 8.0
            try:
                U_new, V_new, W_new, err = _try_rank(n, m, p, rank, T, seed=seed, time_limit=time_limit)
                if err < 1e-10:
                    # Check for exactness with tighter tolerance
                    if _tensor_error(U_new, V_new, W_new, T) < 1e-12:
                        # Polish with gradient descent
                        U_pol, V_pol, W_pol, err_pol = _gradient_descent_polish(
                            U_new, V_new, W_new, T, n, m, p, rank, time_limit=3.0
                        )
                        if err_pol < 1e-12:
                            U, V, W = U_pol, V_pol, W_pol
                            found_rank = rank
                            found_U, found_V, found_W = U, V, W
                            success = True
                            break
            except Exception:
                continue
                
        if success:
            # Found a solution for this rank. Continue to try lower ranks.
            continue
        else:
            # Failed to find solution for this rank.
            # Try the next lower rank only if we have time.
            # If we've already found a rank, keep trying lower.
            # If we haven't found any rank < 50, break after a few failures.
            if found_rank == n * m * p:
                # Haven't found any improvement, break to save time
                break
            else:
                # Already found an improvement, keep trying for better
                continue
    
    # Return the best found
    U_list = found_U.tolist()
    V_list = found_V.tolist()
    W_list = found_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
