"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _compute_error_fast(n, m, p, U, V, W):
    """Fast vectorized error computation for tensor decomposition."""
    rank = U.shape[1]
    # Precompute element-wise products for each factor pair
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build lookup arrays
    err = 0.0
    for i in range(n):
        for k in range(p):
            w_row = W[k * n + i, :]  # shape (rank,)
            for j in range(m):
                u_row = U[i * m + j, :]  # shape (rank,)
                v_row = V[j * p + k, :]  # shape (rank,)
                val = np.dot(u_row, v_row * w_row)
                err += (val - 1.0) ** 2
    return err

def _als_step(n, m, p, U, V, W, rank):
    """One ALS iteration: update W, then U, then V."""
    # Update W: for each (k,i), solve least squares
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
            W[k * n + i, :] = sol
    
    # Update U: for each (i,j), solve least squares
    for i in range(n):
        for j in range(m):
            B = np.zeros((p, rank))
            for k in range(p):
                B[k, :] = V[j * p + k, :] * W[k * n + i, :]
            sol, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
            U[i * m + j, :] = sol
    
    # Update V: for each (j,k), solve least squares
    for j in range(m):
        for k in range(p):
            C = np.zeros((n, rank))
            for i in range(n):
                C[i, :] = U[i * m + j, :] * W[k * n + i, :]
            sol, _, _, _ = np.linalg.lstsq(C, np.ones(n), rcond=None)
            V[j * p + k, :] = sol

def _verify_rank(n, m, p, rank, U, V, W):
    """Verify that the given factors decompose the matmul tensor correctly."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > 1e-6:
                    return False, (i, j, k, val)
    return True, None

def _try_round_and_verify(n, m, p, U, V, W, rank):
    """Try rounding to half-integers and verify."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    valid, _ = _verify_rank(n, m, p, rank, U_r, V_r, W_r)
    if valid:
        return True, U_r, V_r, W_r
    
    # Also try pure integer rounding
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    valid, _ = _verify_rank(n, m, p, rank, U_i, V_i, W_i)
    if valid:
        return True, U_i, V_i, W_i
    
    return False, None, None, None

def _build_block_structure_init(n, m, p, rank):
    """
    Build initialization with block structure.
    For 4x4x6: split into 2 row-blocks of 2 rows each.
    Each row-block uses 'rank/2' terms (if rank is even).
    This gives a structured starting point for ALS.
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Split rows into two groups of 2
    n_half = n // 2
    rank_half = rank // 2
    
    for block in range(2):
        i_start = block * n_half
        r_start = block * rank_half
        r_end = r_start + rank_half
        
        # Random init for this block's terms
        for i in range(i_start, i_start + n_half):
            for j in range(m):
                U[i * m + j, r_start:r_end] = np.random.randn(rank_half) * 0.5
        for j in range(m):
            for k in range(p):
                V[j * p + k, r_start:r_end] = np.random.randn(rank_half) * 0.5
        for k in range(p):
            for i in range(i_start, i_start + n_half):
                W[k * n + i, r_start:r_end] = np.random.randn(rank_half) * 0.5
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s margin out of 150s
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    # Initialize best with schoolbook solution
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank))
    best_V = np.zeros((m * p, schoolbook_rank))
    best_W = np.zeros((p * n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1
                best_V[j * p + k, c] = 1
                best_W[k * n + i, c] = 1
    
    random.seed(42)
    np.random.seed(42)
    
    # Target ranks to try, from most promising to least
    # Known: rank(2x4x3) <= 6, so rank(4x4x6) <= 12 (two independent 2x4x3 blocks)
    # Try: 12, 14, 16, 18, 20, 24
    target_ranks = [12, 14, 16, 18, 20, 24]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        max_attempts = 15
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt % 3 == 0:
                # Block-structured initialization
                U, V, W = _build_block_structure_init(n, m, p, target_rank)
            elif attempt % 3 == 1:
                # Random initialization
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
            else:
                # Sparse random initialization
                U = np.random.randn(n * m, target_rank) * 0.3
                V = np.random.randn(m * p, target_rank) * 0.3
                W = np.random.randn(p * n, target_rank) * 0.3
                # Zero out some entries to encourage sparsity
                for c in range(target_rank):
                    mask_u = np.random.random(n * m) < 0.3
                    mask_v = np.random.random(m * p) < 0.3
                    mask_w = np.random.random(p * n) < 0.3
                    U[mask_u, c] = 0
                    V[mask_v, c] = 0
                    W[mask_w, c] = 0
            
            # ALS iterations
            prev_err = float('inf')
            for it in range(300):
                if time.time() > deadline:
                    break
                
                err = _compute_error_fast(n, m, p, U, V, W)
                
                if err < 1e-10:
                    ok, U_r, V_r, W_r = _try_round_and_verify(n, m, p, U, V, W, target_rank)
                    if ok:
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
                
                # Check for convergence
                if abs(prev_err - err) < 1e-14:
                    # Try rounding before giving up
                    ok, U_r, V_r, W_r = _try_round_and_verify(n, m, p, U, V, W, target_rank)
                    if ok:
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
                    break
                
                prev_err = err
                _als_step(n, m, p, U, V, W, target_rank)
            
            if found:
                break
        else:
            # If we exhausted attempts for this rank, continue to next
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
