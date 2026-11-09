"""Program that searches for a lower-rank tensor decomposition of matrix multiplication 3x5x7."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n=3, m=5, p=7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 5, 7
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D arrays
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_decomposition_fast(U, V, W, n=3, m=5, p=7):
    """Fast validation using numpy einsum."""
    n, m, p = 3, 5, 7
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # Use einsum: sum over r
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.allclose(T, 1.0, atol=1e-9)

def _try_rank_reduction(U, V, W, n=3, m=5, p=7, max_iter=500, tol=1e-10):
    """Try to reduce rank by finding dependent columns and eliminating them."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # For each column, check if it's linearly dependent on others in a way that
    # preserves the tensor equation. This is complex, so we use a different approach.
    # Try removing one column at a time and see if the remaining can be adjusted.
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    for remove_col in range(rank - 1, -1, -1):
        # Try removing column remove_col
        U_test = np.delete(U_arr, remove_col, axis=1)
        V_test = np.delete(V_arr, remove_col, axis=1)
        W_test = np.delete(W_arr, remove_col, axis=1)
        
        # Check if the remaining factors still produce the correct tensor
        if _validate_decomposition_fast(
            U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p
        ):
            best_rank = U_test.shape[1]
            best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
            # Continue trying to remove more
            U_arr, V_arr, W_arr = U_test, V_test, W_test
            rank = U_arr.shape[1]
            continue
    
    return best_U, best_V, best_W, best_rank

def _greedy_construct(n=3, m=5, p=7):
    """Construct a decomposition using a greedy approach."""
    # Start with schoolbook
    base_rank = n * m * p
    U = np.zeros((n * m, base_rank))
    V = np.zeros((m * p, base_rank))
    W = np.zeros((p * n, base_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _local_search(U, V, W, n=3, m=5, p=7, max_time=30, step_size=0.1):
    """Local search to reduce the 'effective rank' by making some columns zero."""
    start = time.time()
    deadline = start + max_time
    
    U_arr = np.array(U, dtype=float).copy()
    V_arr = np.array(V, dtype=float).copy()
    W_arr = np.array(W, dtype=float).copy()
    rank = U_arr.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Try to push some columns toward zero
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Pick a random column to try to eliminate
        col = random.randint(0, rank - 1)
        
        # Compute the current contribution of this column
        # We want to see if we can redistribute this column's contribution
        # to other columns
        
        # For now, just try small perturbations
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Perturb the column slightly
            U_pert = U_arr.copy()
            V_pert = V_arr.copy()
            W_pert = W_arr.copy()
            
            # Add a small random perturbation
            noise_U = np.random.randn(n * m) * step_size
            noise_V = np.random.randn(m * p) * step_size
            noise_W = np.random.randn(p * n) * step_size
            
            U_pert[:, col] += noise_U
            V_pert[:, col] += noise_V
            W_pert[:, col] += noise_W
            
            # Check if this is closer to having a zero column
            col_norm = np.linalg.norm(U_pert[:, col]) + np.linalg.norm(V_pert[:, col]) + np.linalg.norm(W_pert[:, col])
            
            if col_norm < 1e-6:
                # This column is effectively zero, we can remove it
                U_new = np.delete(U_pert, col, axis=1)
                V_new = np.delete(V_pert, col, axis=1)
                W_new = np.delete(W_pert, col, axis=1)
                
                if _validate_decomposition_fast(
                    U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p
                ):
                    best_rank = U_new.shape[1]
                    best_U, best_V, best_W = U_new, V_new, W_new
                    U_arr, V_arr, W_arr = U_new, V_new, W_new
                    rank = U_arr.shape[1]
                    break
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook
    U_school, V_school, W_school = _greedy_construct(n, m, p)
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = base_rank
    
    # Try to reduce rank by removing dependent columns
    if time.time() < deadline - 20:
        U_red, V_red, W_red, rank_red = _try_rank_reduction(
            U_school.tolist(), V_school.tolist(), W_school.tolist(), n, m, p
        )
        if rank_red < best_rank:
            best_rank = rank_red
            best_result = {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
    
    # Try local search
    if time.time() < deadline - 10:
        U_ls, V_ls, W_ls, rank_ls = _local_search(
            best_result["U"], best_result["V"], best_result["W"], n, m, p,
            max_time=min(30, deadline - time.time() - 5)
        )
        if rank_ls < best_rank:
            best_rank = rank_ls
            best_result = {"U": U_ls.tolist(), "V": V_ls.tolist(), "W": W_ls.tolist()}
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above. This block is kept for structural compatibility.
# EVOLVE-BLOCK-END
