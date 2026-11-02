"""Matrix multiplication 3x4x5 - rank-minimal exact tensor decomposition search."""
import time
import itertools
import random
import sys

try:
    import numpy as np
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DIMENSIONS = [3, 4, 5]

def _build_schoolbook():
    """Return the schoolbook (rank=60) decomposition as a baseline."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is exact.
    The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k)
    must equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    """
    rank = len(U[0])
    # Build matrices for fast verification
    U_mat = np.array(U, dtype=float)  # (n*m, rank)
    V_mat = np.array(V, dtype=float)  # (m*p, rank)
    W_mat = np.array(W, dtype=float)  # (p*n, rank)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_mat[i * m + j]
                v_row = V_mat[j * p + k]
                w_row = W_mat[k * n + i]
                val = np.sum(u_row * v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Faster verification using tensor contraction.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    """
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    rank = U_mat.shape[1]
    
    # Reshape: U is (n,m,rank), V is (m,p,rank), W is (p,n,rank)
    U_reshaped = U_mat.reshape(n, m, rank)
    V_reshaped = V_mat.reshape(m, p, rank)
    W_reshaped = W_mat.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # We can compute this as: for each c, outer product contribution
    # T = sum_c np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        T += U_reshaped[:, :, c][:, :, np.newaxis] * V_reshaped[:, c, :][np.newaxis, :, :] * W_reshaped[c, :, :][:, np.newaxis, :]
    
    # Wait, the indices need to be correct.
    # U_reshaped[i,j,c] -> shape (n, m, rank)
    # V_reshaped[j,k,c] -> shape (m, p, rank)  
    # W_reshaped[k,i,c] -> shape (p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Let me redo this more carefully
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        # U[:, :, c] has shape (n, m)
        # V[:, :, c] has shape (m, p)  -- wait, V_reshaped is (m, p, rank), so V_reshaped[:,:,c] is (m,p)
        # W[:, :, c] has shape (p, n)  -- W_reshaped is (p, n, rank), so W_reshaped[:,:,c] is (p,n)
        
        # T[i,j,k] += U[i,j] * V[j,k] * W[k,i]
        # This is: for each (i,j,k), U[i,j]*V[j,k]*W[k,i]
        # = einsum('ij,jk,ki->ijk', U[:,:,c], V[:,:,c], W[:,:,c])
        T += np.einsum('ij,jk,ki->ijk', U_reshaped[:, :, c], V_reshaped[:, :, c], W_reshaped[:, :, c])
    
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def _try_rank_with_optimization(rank, n, m, p, time_budget=30, seed=42):
    """
    Try to find a decomposition of given rank using numerical optimization.
    Returns (U, V, W) if found, else None.
    """
    if not HAS_SCIPY:
        return None
    
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # Parameterize: U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Total parameters: (n*m + m*p + p*n) * rank
    n_params = (n * m + m * p + p * n) * rank
    
    def objective(params):
        U_flat = params[:n * m * rank]
        V_flat = params[n * m * rank : n * m * rank + m * p * rank]
        W_flat = params[n * m * rank + m * p * rank :]
        
        U_mat = U_flat.reshape(n * m, rank)
        V_mat = V_flat.reshape(m * p, rank)
        W_mat = W_flat.reshape(p * n, rank)
        
        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)
        
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U_reshaped[:, :, c], V_reshaped[:, :, c], W_reshaped[:, :, c])
        
        # Minimize ||T - ones||_F^2
        diff = T - 1.0
        return np.sum(diff * diff)
    
    # Try multiple random restarts
    best_params = None
    best_val = float('inf')
    
    start_time = time.time()
    max_restarts = 10
    attempts = 0
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with random values
        params0 = np.random.randn(n_params) * 0.5
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                if best_val < 1e-6:
                    break
        except Exception:
            continue
        
        attempts += 1
    
    if best_params is None or best_val > 1e-4:
        return None
    
    # Extract and verify
    U_flat = best_params[:n * m * rank]
    V_flat = best_params[n * m * rank : n * m * rank + m * p * rank]
    W_flat = best_params[n * m * rank + m * p * rank :]
    
    U_mat = U_flat.reshape(n * m, rank)
    V_mat = V_flat.reshape(m * p, rank)
    W_mat = W_flat.reshape(p * n, rank)
    
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    if _verify_decomposition_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return U_list, V_list, W_list
    
    return None

def _try_lanczos_like(rank, n, m, p, time_budget=20, seed=42):
    """
    Try a Lanczos-like iterative approach to find a low-rank decomposition.
    """
    if not HAS_SCIPY:
        return None
    
    rng = random.Random(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Initialize with random matrices
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    lr = 0.1
    for iteration in range(1000):
        if time.time() - start_time > time_budget:
            break
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U_reshaped[:, :, c], V_reshaped[:, :, c], W_reshaped[:, :, c])
        
        diff = T - 1.0
        loss = np.sum(diff * diff)
        
        if loss < 1e-10:
            break
        
        # Compute gradients
        # dL/dU[i,j,c] = 2 * sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
        dU = np.zeros_like(U_reshaped)
        dV = np.zeros_like(V_reshaped)
        dW = np.zeros_like(W_reshaped)
        
        for c in range(rank):
            # dU[i,j,c] = 2 * sum_k diff[i,j,k] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
            dU[:, :, c] = 2 * np.einsum('ijk,jk,ki->ij', diff, V_reshaped[:, :, c], W_reshaped[:, :, c])
            # dV[j,k,c] = 2 * sum_i diff[i,j,k] * U_reshaped[i,j,c] * W_reshaped[k,i,c]
            dV[:, :, c] = 2 * np.einsum('ijk,ij,ki->jk', diff, U_reshaped[:, :, c], W_reshaped[:, :, c])
            # dW[k,i,c] = 2 * sum_j diff[i,j,k] * U_reshaped[i,j,c] * V_reshaped[j,k,c]
            dW[:, :, c] = 2 * np.einsum('ijk,ij,jk->ki', diff, U_reshaped[:, :, c], V_reshaped[:, :, c])
        
        U -= lr * dU.reshape(n * m, rank)
        V -= lr * dV.reshape(m * p, rank)
        W -= lr * dW.reshape(p * n, rank)
        
        # Decay learning rate
        lr *= 0.999
    
    loss = objective(U.reshape(-1)) if False else None
    # Final check
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return U_list, V_list, W_list
    
    return None

def _search_random_rank(rank, n, m, p, time_budget=15, seed=42):
    """
    Try random initializations with gradient descent for a given rank.
    """
    if not HAS_SCIPY:
        return None
    
    np.random.seed(seed)
    start_time = time.time()
    
    n_params = (n * m + m * p + p * n) * rank
    
    def objective(params):
        U_flat = params[:n * m * rank]
        V_flat = params[n * m * rank : n * m * rank + m * p * rank]
        W_flat = params[n * m * rank + m * p * rank :]
        
        U_mat = U_flat.reshape(n * m, rank)
        V_mat = V_flat.reshape(m * p, rank)
        W_mat = W_flat.reshape(p * n, rank)
        
        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)
        
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            T += np.einsum('ij,jk,ki->ijk', U_reshaped[:, :, c], V_reshaped[:, :, c], W_reshaped[:, :, c])
        
        diff = T - 1.0
        return np.sum(diff * diff)
    
    best_params = None
    best_val = float('inf')
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        params0 = np.random.randn(n_params) * 1.0
        
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                if best_val < 1e-8:
                    break
        except Exception:
            continue
    
    if best_params is None or best_val > 1e-6:
        return None
    
    U_flat = best_params[:n * m * rank]
    V_flat = best_params[n * m * rank : n * m * rank + m * p * rank]
    W_flat = best_params[n * m * rank + m * p * rank :]
    
    U_mat = U_flat.reshape(n * m, rank)
    V_mat = V_flat.reshape(m * p, rank)
    W_mat = W_flat.reshape(p * n, rank)
    
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    if _verify_decomposition_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return U_list, V_list, W_list
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Known lower bound: at least n*m = 12 (for square it's ~n^2.5, but for 3x4x5)
    # The minimal rank is unknown but bounded. Let's search from low to high.
    # Known results: rank(M_{3,3,3}) <= 23 (Strassen improved to 23, best known 23)
    # For 3x4x5, let's try ranks from 40 upwards (being conservative)
    
    # First, get the schoolbook as fallback
    fallback = _build_schoolbook()
    
    # Try to find better decompositions
    # Start from a reasonable lower bound and go up
    # Theoretical lower bound: max(n*m, m*p, p*n) / something... 
    # Actually, a simple lower bound is max(n*m, m*p, p*n) = 20
    # But that's not tight. Let's try from 35 to 55.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    # Try various ranks, starting from lower
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from 35 to 59
    for rank in range(35, min(60, schoolbook_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try different methods for this rank
        for seed in [42, 123, 777, 2024, 999]:
            if time.time() > deadline - 10:
                break
            
            # Try optimization-based approach
            result = _try_rank_with_optimization(rank, n, m, p, time_budget=10, seed=seed)
            if result is not None:
                U, V, W = result
                if best_result is None or rank < best_rank:
                    best_result = (U, V, W)
                    best_rank = rank
                    break  # Found for this rank, try next lower rank
        
        if best_result is not None and best_rank == rank:
            # Found a decomposition at this rank, try lower
            continue
        elif best_result is not None:
            # Found at a higher rank, continue trying lower
            continue
        else:
            # Not found at this rank, continue
            continue
    
    # If we found something better, return it
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    return fallback
# EVOLVE-BLOCK-END
