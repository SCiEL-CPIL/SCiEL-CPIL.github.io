import time
import sys
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _build_tensor_indices(n, m, p):
    """Return list of (u_idx, v_idx, w_idx) for each (i,j,k)."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    return triples

def _compute_residual(U, V, W, n, m, p):
    """Compute residual tensor R[i,j,k] = 1 - sum_r U[i,j,r]V[j,k,r]W[k,i,r]."""
    r = U.shape[1]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # T_calc[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T_calc = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    return np.ones((n, m, p), dtype=float) - T_calc

def _exact_rank_solution(n, m, p, rank, rng, time_limit=10.0):
    """Attempt to find an exact integer decomposition of given rank."""
    N = n * m
    P = m * p
    Q = p * n
    
    start_time = time.time()
    max_iters = 1000
    
    for attempt in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = rng.integers(-1, 3, size=(N, rank))
        V = rng.integers(-1, 3, size=(P, rank))
        W = rng.integers(-1, 3, size=(Q, rank))
        
        # Local search
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
                
            R = _compute_residual(U, V, W, n, m, p)
            max_err = np.max(np.abs(R))
            
            if max_err < 1e-9:
                return (U.astype(int), V.astype(int), W.astype(int))
            
            # Find worst element
            ri, rj, rk = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            
            # Try to fix it by adjusting one variable
            # Deficit = 1 - current_sum
            current_sum = np.sum(U[ri, rj, :] * V[rj, rk, :] * W[rk, ri, :])
            deficit = 1.0 - current_sum
            
            if abs(deficit) < 1e-9:
                continue
                
            # Pick a random rank index
            r_idx = rng.integers(0, rank)
            
            u_val = U[ri, rj, r_idx]
            v_val = V[rj, rk, r_idx]
            w_val = W[rk, ri, r_idx]
            
            # If product of other two is 0, we can't fix it by changing this one easily
            if v_val * w_val == 0:
                continue
                
            # delta = deficit / (v_val * w_val)
            delta = deficit / (v_val * w_val)
            new_u_val = u_val + delta
            
            # Round to nearest integer
            rounded = int(round(new_u_val))
            
            if rounded != u_val:
                # Check if this step reduces error
                old_U = U[ri, rj, r_idx]
                U[ri, rj, r_idx] = rounded
                
                new_R = _compute_residual(U, V, W, n, m, p)
                new_max_err = np.max(np.abs(new_R))
                
                if new_max_err >= max_err:
                    U[ri, rj, r_idx] = old_U
                else:
                    # Keep the change
                    pass
        # End local search
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0
    
    rng = np.random.default_rng(12345)
    
    # Try ranks from 44 down to 41
    # We allocate time based on remaining budget
    for rank in [44, 43, 42, 41]:
        if time.time() >= deadline:
            break
        # Give each rank attempt a fraction of the remaining time
        # But ensure we don't exceed deadline
        time_left = deadline - time.time()
        if time_left < 5:
            break
        # Use 20% of remaining time for each rank attempt, capped at 20s
        t_limit = min(20.0, time_left * 0.2)
        
        sol = _exact_rank_solution(n, m, p, rank, rng, time_limit=t_limit)
        if sol is not None:
            U, V, W = sol
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
            }

    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
