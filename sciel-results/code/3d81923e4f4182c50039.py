"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _verify_rank(n, m, p, rank, U, V, W, tol=1e-6):
    """Verify that the given factors decompose the matmul tensor correctly."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def _update_W(U, V, W, n, m, p, r):
    """Update W rows given U and V."""
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, r))
            for j in range(m):
                A[j, :] = U[i*m + j, :] * V[j*p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k*n + i, :] = sol

def _update_U(U, V, W, n, m, p, r):
    """Update U rows given V and W."""
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, r))
            for k in range(p):
                B[k, :] = V[j*p + k, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i*m + j, :] = sol

def _update_V(U, V, W, n, m, p, r):
    """Update V rows given U and W."""
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, r))
            for i in range(n):
                C[i, :] = U[i*m + j, :] * W[k*n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j*p + k, :] = sol

def _compute_error(U, V, W, n, m, p, r):
    """Compute the sum of squared errors for the decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                err += (val - 1.0)**2
    return err

def _als_optimize(U, V, W, n, m, p, r, max_iters=200, tol=1e-8, deadline=None):
    """Run ALS optimization with proper loop structure."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < tol:
            break
        
        _update_W(U, V, W, n, m, p, r)
        _update_U(U, V, W, n, m, p, r)
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def _polish_solution(U, V, W, n, m, p, r, deadline=None):
    """Final high-precision polish pass."""
    for it in range(100):
        if deadline is not None and time.time() > deadline:
            break
        
        err = _compute_error(U, V, W, n, m, p, r)
        if err < 1e-12:
            break
        
        _update_W(U, V, W, n, m, p, r)
        _update_U(U, V, W, n, m, p, r)
        _update_V(U, V, W, n, m, p, r)
    
    return U, V, W

def _is_half_integer(mat, tol=1e-4):
    """Check if all entries are close to multiples of 0.5."""
    rounded = np.round(mat * 2) / 2
    return np.allclose(mat, rounded, atol=tol)

def _round_to_half_integer(mat):
    return np.round(mat * 2) / 2

def _construct_schoolbook(n, m, p):
    """Construct the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    return U, V, W, rank

def _construct_block_decomp(n, m, p, block_n, block_p, block_rank, U_sub, V_sub, W_sub):
    """
    Construct a block-embedded decomposition for n x m * m x p.
    Partition rows of A into blocks of size block_n, columns of B into blocks of size block_p.
    For each block pair (ib, kb), embed the sub-decomposition.
    
    The sub-decomp handles: A[ib*block_n:(ib+1)*block_n, :] * B[:, kb*block_p:(kb+1)*block_p]
    producing C[ib*block_n:(ib+1)*block_n, kb*block_p:(kb+1)*block_p]
    
    U_sub: shape (block_n * m, block_rank)  -- rows indexed by (i_sub, j)
    V_sub: shape (m * block_p, block_rank)  -- rows indexed by (j, k_sub)
    W_sub: shape (block_p * block_n, block_rank) -- rows indexed by (k_sub, i_sub)
    """
    n_blocks = n // block_n
    p_blocks = p // block_p
    total_rank = n_blocks * p_blocks * block_rank
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for ib in range(n_blocks):
        for kb in range(p_blocks):
            i_start = ib * block_n
            k_start = kb * block_p
            c_start = (ib * p_blocks + kb) * block_rank
            
            # Embed U_sub: for i in 0..block_n-1, j in 0..m-1
            for i in range(block_n):
                for j in range(m):
                    U[i_start + i, c_start:c_start+block_rank] = U_sub[i*m + j, :]
            
            # Embed V_sub: for j in 0..m-1, k in 0..block_p-1
            for j in range(m):
                for k in range(block_p):
                    V[j*p + k_start + k, c_start:c_start+block_rank] = V_sub[j*block_p + k, :]
            
            # Embed W_sub: for k in 0..block_p-1, i in 0..block_n-1
            for k in range(block_p):
                for i in range(block_n):
                    W[k_start + k, c_start:c_start+block_rank] = W_sub[k*block_n + i, :]
    
    return U, V, W, total_rank

def _find_sub_decomp(n_sub, m_sub, p_sub, target_rank, deadline, max_attempts=15):
    """
    Try to find a decomposition for n_sub x m_sub * m_sub x p_sub with given rank.
    Returns (U_sub, V_sub, W_sub) or None.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(attempt * 1000 + 42)
        scale = 0.3 + 0.3 * (attempt % 4)
        U_sub = rng.randn(n_sub * m_sub, target_rank) * scale
        V_sub = rng.randn(m_sub * p_sub, target_rank) * scale
        W_sub = rng.randn(p_sub * n_sub, target_rank) * scale
        
        # Normalize each column
        for c in range(target_rank):
            norm = np.sqrt(np.sum(U_sub[:, c]**2) * np.sum(V_sub[:, c]**2) * np.sum(W_sub[:, c]**2))
            if norm > 1e-10:
                scale_c = 1.0 / (norm ** (1.0/3.0))
                U_sub[:, c] *= scale_c
                V_sub[:, c] *= scale_c
                W_sub[:, c] *= scale_c
        
        U_sub, V_sub, W_sub = _als_optimize(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, target_rank, 
                                              max_iters=150, deadline=deadline)
        
        err = _compute_error(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, target_rank)
        
        if err < 1e-6:
            # Try to round to half-integers
            if _is_half_integer(U_sub) and _is_half_integer(V_sub) and _is_half_integer(W_sub):
                U_r = _round_to_half_integer(U_sub)
                V_r = _round_to_half_integer(V_sub)
                W_r = _round_to_half_integer(W_sub)
                valid, _ = _verify_rank(n_sub, m_sub, p_sub, target_rank, U_r, V_r, W_r)
                if valid:
                    return U_r, V_r, W_r
            else:
                # Try polishing to see if it converges to half-integers
                U_sub, V_sub, W_sub = _polish_solution(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, 
                                                        target_rank, deadline=deadline)
                err = _compute_error(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, target_rank)
                if err < 1e-8:
                    if _is_half_integer(U_sub) and _is_half_integer(V_sub) and _is_half_integer(W_sub):
                        U_r = _round_to_half_integer(U_sub)
                        V_r = _round_to_half_integer(V_sub)
                        W_r = _round_to_half_integer(W_sub)
                        valid, _ = _verify_rank(n_sub, m_sub, p_sub, target_rank, U_r, V_r, W_r)
                        if valid:
                            return U_r, V_r, W_r
                    else:
                        # Return unrounded if valid
                        valid, _ = _verify_rank(n_sub, m_sub, p_sub, target_rank, U_sub, V_sub, W_sub)
                        if valid:
                            return U_sub, V_sub, W_sub
    
    return None

def _try_direct_search(n, m, p, target_rank, deadline, max_attempts=8):
    """
    Try to directly find a decomposition for n x m * m x p with given rank.
    Returns (U, V, W) or None.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(attempt * 7919 + 13)
        scale = 0.2 + 0.3 * (attempt % 4)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
        
        for c in range(target_rank):
            norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
            if norm > 1e-10:
                scale_c = 1.0 / (norm ** (1.0/3.0))
                U[:, c] *= scale_c
                V[:, c] *= scale_c
                W[:, c] *= scale_c
        
        U, V, W = _als_optimize(U, V, W, n, m, p, target_rank, max_iters=100, deadline=deadline)
        
        err = _compute_error(U, V, W, n, m, p, target_rank)
        
        if err < 1e-6:
            if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                U_r = _round_to_half_integer(U)
                V_r = _round_to_half_integer(V)
                W_r = _round_to_half_integer(W)
                valid, _ = _verify_rank(n, m, p, target_rank, U_r, V_r, W_r)
                if valid:
                    return U_r, V_r, W_r
            else:
                U, V, W = _polish_solution(U, V, W, n, m, p, target_rank, deadline=deadline)
                err = _compute_error(U, V, W, n, m, p, target_rank)
                if err < 1e-8:
                    if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                        U_r = _round_to_half_integer(U)
                        V_r = _round_to_half_integer(V)
                        W_r = _round_to_half_integer(W)
                        valid, _ = _verify_rank(n, m, p, target_rank, U_r, V_r, W_r)
                        if valid:
                            return U_r, V_r, W_r
                    else:
                        valid, _ = _verify_rank(n, m, p, target_rank, U, V, W)
                        if valid:
                            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s margin out of 120s
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best_U, best_V, best_W, best_rank = _construct_schoolbook(n, m, p)
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Block decomposition using 2x4 * 4x3 sub-blocks
    # For 4x4 * 4x6: split into 2 blocks in n (size 2), 2 blocks in p (size 3)
    # Sub-problem: 2x4 * 4x3, rank target: try 8, 9, 10, 11, 12
    # Total rank = 2 * 2 * sub_rank = 4 * sub_rank
    
    # Try to find sub-decompositions for 2x4 * 4x3
    sub_n, sub_m, sub_p = 2, 4, 3
    
    for sub_rank in range(12, 7, -1):
        if time.time() > deadline - 10:
            break
        
        result = _find_sub_decomp(sub_n, sub_m, sub_p, sub_rank, deadline - 10, max_attempts=12)
        if result is not None:
            U_sub, V_sub, W_sub = result
            U, V, W, total_rank = _construct_block_decomp(n, m, p, sub_n, sub_p, sub_rank, U_sub, V_sub, W_sub)
            
            valid, _ = _verify_rank(n, m, p, total_rank, U, V, W)
            if valid:
                best_rank = total_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Strategy 2: Try direct search for ranks below current best
    for target_rank in range(best_rank - 1, max(best_rank - 15, 15), -1):
        if time.time() > deadline - 5:
            break
        
        result = _try_direct_search(n, m, p, target_rank, deadline - 5, max_attempts=5)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
