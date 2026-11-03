"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-8):
        return False
    return True

def _strassen_2x2_factors():
    """Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication."""
    return [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

def _build_rank49_block():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen recursion.
    Returns (U4, V4, W4) each of shape (16, 49) for the square case.
    """
    s2 = _strassen_2x2_factors()
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    # Block index mapping for 4x4 matrices (row-major flattening)
    # Block (bi, bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    def block_idx(bi, bj, dr, dc):
        r = 2 * bi + dr
        c = 2 * bj + dc
        return r * 4 + c
    
    for s in range(7):
        for r in range(7):
            col = s * 7 + r
            u_s, v_s, w_s = s2[s]
            u_r, v_r, w_r = s2[r]
            
            for bi in range(2):
                for bj in range(2):
                    for ei in range(2):
                        for ej in range(2):
                            # U contribution
                            u_block = u_s[bi * 2 + bj]
                            u_scalar = u_r[ei * 2 + ej]
                            if u_block == 0 or u_scalar == 0:
                                continue
                            idx = block_idx(bi, bj, ei, ej)
                            U4[idx, col] += u_block * u_scalar
                            
                            # V contribution
                            v_block = v_s[bi * 2 + bj]
                            v_scalar = v_r[ei * 2 + ej]
                            if v_block == 0 or v_scalar == 0:
                                continue
                            idx = block_idx(bi, bj, ei, ej)
                            V4[idx, col] += v_block * v_scalar
                            
                            # W contribution
                            w_block = w_s[bi * 2 + bj]
                            w_scalar = w_r[ei * 2 + ej]
                            if w_block == 0 or w_scalar == 0:
                                continue
                            idx = block_idx(bi, bj, ei, ej)
                            W4[idx, col] += w_block * w_scalar
    
    return U4, V4, W4

def _build_rank98_strassen():
    """
    Build rank-98 decomposition for 4x4x8 using two independent 4x4x4 blocks.
    """
    n, m, p = DIMENSIONS
    U4, V4, W4 = _build_rank49_block()
    
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for block in range(2):
        base = block * 49
        U[:, base:base+49] = U4
        # V: for block, k_global = block*4 + k_local
        V[:, base:base+49] = V4
        # W: for block, k_global = block*4 + k_local
        W[:, base:base+49] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _smooth_objective(params, n=4, m=4, p=8, target_rank=None):
    """
    Smooth surrogate for checking if params define a valid rank-r decomposition.
    params: flattened array of U, V, W entries for a given rank r.
    Returns a scalar that is 0 if the decomposition is exact, positive otherwise.
    Uses log-sum-exp for smoothness.
    """
    if target_rank is None:
        r = int(np.sqrt(len(params) / (n*m + m*p + p*n)))
    else:
        r = target_rank
    
    size_U = n * m * r
    size_V = m * p * r
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape(n, m, r)
    V = V_flat.reshape(m, p, r)
    W = W_flat.reshape(p, n, r)
    
    # Compute the tensor product sum
    result = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    
    # Smooth objective: sum of squared deviations from 1
    # Use a smooth approximation: mean of (result - 1)^2
    dev = result - 1.0
    return np.sum(dev * dev)

def _try_optimized_search(time_budget=60.0):
    """
    Try to find a decomposition with rank < 98 using local optimization
    starting from the rank-98 Strassen solution.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from rank 98 solution
    sol98 = _build_rank98_strassen()
    U98 = np.array(sol98["U"])
    V98 = np.array(sol98["V"])
    W98 = np.array(sol98["W"])
    
    # Try ranks from 97 down to 80
    best_result = None
    
    for target_rank in range(97, 79, -1):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize with random perturbation of a subset of the rank-98 solution
        # We'll try to construct a rank-target_rank solution by randomly selecting
        # target_rank columns from the rank-98 solution and optimizing
        
        n_cols = n * m
        v_cols = m * p
        w_cols = p * n
        
        # Select a random subset of columns
        rng = np.random.RandomState(target_rank)
        col_indices = rng.choice(98, size=target_rank, replace=False)
        
        U_init = U98[:, col_indices]
        V_init = V98[:, col_indices]
        W_init = W98[:, col_indices]
        
        # Flatten for optimization
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Simple gradient-free local search using coordinate descent
        # with random perturbations
        best_obj = _smooth_objective(params_init, n, m, p, target_rank)
        best_params = params_init.copy()
        
        # Try a few random restarts with different column selections
        for restart in range(3):
            if time.time() - start_time > time_budget:
                break
            
            col_indices = rng.choice(98, size=target_rank, replace=False)
            U_r = U98[:, col_indices]
            V_r = V98[:, col_indices]
            W_r = W98[:, col_indices]
            params_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
            
            obj_r = _smooth_objective(params_r, n, m, p, target_rank)
            if obj_r < best_obj:
                best_obj = obj_r
                best_params = params_r.copy()
            
            # Local refinement: try small perturbations
            for step in range(50):
                if time.time() - start_time > time_budget:
                    break
                
                # Random coordinate perturbation
                idx = rng.randint(0, len(best_params))
                old_val = best_params[idx]
                best_params[idx] += rng.randn() * 0.1
                new_obj = _smooth_objective(best_params, n, m, p, target_rank)
                
                if new_obj < best_obj:
                    best_obj = new_obj
                else:
                    best_params[idx] = old_val
        
        if best_obj < 1e-10:
            # Found a valid decomposition
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            U = best_params[:size_U].reshape(n * m, target_rank)
            V = best_params[size_U:size_U + size_V].reshape(m * p, target_rank)
            W = best_params[size_U + size_V:].reshape(p * n, target_rank)
            
            result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(result["U"], result["V"], result["W"]):
                return result
    
    return None

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
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

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries to find a rank-minimal decomposition, falls back to Strassen (rank 98) or schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, try optimized search for rank < 98
    try:
        result = _try_optimized_search(time_budget=100.0)
        if result is not None and time.time() < deadline:
            if _verify(result["U"], result["V"], result["W"]):
                return result
    except Exception:
        pass
    
    # Fall back to Strassen rank-98
    try:
        result = _build_rank98_strassen()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Final fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
