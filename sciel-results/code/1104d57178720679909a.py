import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W form a valid tensor decomposition of the n x m x p matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Trace convention: W[k*n + i, r]
    """
    T = np.ones((n, m, p))
    
    T_decomp = np.zeros((n, m, p))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    if not np.allclose(T, T_decomp, atol=1e-9):
        return False
    return True

def _schoolbook():
    """Generate the standard schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_rank_reduction(U_init, V_init, W_init, n, m, p, target_rank, max_iters=100, tol=1e-6):
    """
    Try to reduce rank by iteratively removing columns that are linear combinations
    of others, using a greedy approach.
    """
    U = np.array(U_init, dtype=float).copy()
    V = np.array(V_init, dtype=float).copy()
    W = np.array(W_init, dtype=float).copy()
    
    current_rank = U.shape[1]
    
    for iteration in range(max_iters):
        if current_rank <= target_rank:
            break
        
        # Try removing each column and check if the decomposition still holds
        best_remove = -1
        best_error = np.inf
        
        for r in range(current_rank):
            # Create masks to exclude column r
            mask = [c for c in range(current_rank) if c != r]
            
            if len(mask) < target_rank:
                continue
            
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            
            # Check if this sub-decomposition can still represent the full tensor
            # by checking if the removed column is in the span of the others
            # For simplicity, we'll just verify if the sub-decomposition works
            # (This is expensive but correct)
            
            T_decomp = np.zeros((n, m, p))
            for idx, r_sub in enumerate(mask):
                u_col = U[:, r_sub]
                v_col = V[:, r_sub]
                w_col = W[:, r_sub]
                
                u_2d = u_col.reshape(n, m)
                v_2d = v_col.reshape(m, p)
                w_2d = w_col.reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_decomp[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
            
            error = np.max(np.abs(T_decomp - 1.0))
            
            if error < best_error:
                best_error = error
                best_remove = r
        
        if best_remove >= 0 and best_error < tol:
            mask = [c for c in range(current_rank) if c != best_remove]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            current_rank -= 1
        else:
            break
    
    if current_rank <= target_rank:
        return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank], target_rank
    return None, None, None, current_rank

def _structured_search(n, m, p, max_rank_to_try, time_limit):
    """
    Try structured rank reductions using the algebraic properties of matmul.
    """
    start_time = time.time()
    
    # Start from schoolbook
    U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank incrementally
    for target_rank in range(rank - 1, max(0, rank - 20), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random perturbations of the schoolbook decomposition
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            
            # Create a slightly perturbed version
            U_pert = U.copy()
            V_pert = V.copy()
            W_pert = W.copy()
            
            # Add small random noise to encourage different linear dependencies
            noise_level = 1e-3
            U_pert += noise_level * np.random.randn(n * m, rank)
            V_pert += noise_level * np.random.randn(m * p, rank)
            W_pert += noise_level * np.random.randn(p * n, rank)
            
            # Try to reduce rank from this perturbed version
            U_reduced, V_reduced, W_reduced, new_rank = _try_rank_reduction(
                U_pert, V_pert, W_pert, n, m, p, target_rank, max_iters=50, tol=1e-5
            )
            
            if U_reduced is not None and new_rank < best_rank:
                # Verify the result
                if _verify(U_reduced.tolist(), V_reduced.tolist(), W_reduced.tolist(), n, m, p, new_rank):
                    best_U, best_V, best_W, best_rank = U_reduced, V_reduced, W_reduced, new_rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 3x5x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook (should always pass)
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
        pass
    
    # Try structured search for rank reduction
    best_U, best_V, best_W, best_rank = _structured_search(n, m, p, 70, 120.0)
    
    # If we found something better, use it; otherwise fall back to schoolbook
    if best_rank < rank_schoolbook:
        U, V, W, rank = best_U, best_V, best_W, best_rank
    
    # Final verification
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank):
        # Fall back to schoolbook if verification fails
        U, V, W, rank = _schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
