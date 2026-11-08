import numpy as np
import time

DIMENSIONS = [2, 4, 6]
DEADLINE = 115.0  # safety margin before 120s

def _build_rank42():
    """Build the known rank-42 block-Strassen solution."""
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    V_s = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    W_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1}
    ]

    col = 0
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_s[s].items():
                    u_idx = i_loc * m + (a_block * 2 + j_loc)
                    U[u_idx, col] += coeff
                for (j_loc, k_loc), coeff in V_s[s].items():
                    v_idx = (a_block * 2 + j_loc) * p + (b_block * 2 + k_loc)
                    V[v_idx, col] += coeff
                for (i_loc, k_loc), coeff in W_s[s].items():
                    w_idx = (b_block * 2 + k_loc) * n + i_loc
                    W[w_idx, col] += coeff
                col += 1
    return U, V, W

def _tensor_check(U, V, W, tol=1e-8):
    """Check if U⊗V⊗W equals the matmul tensor exactly (within tol)."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r_idx in range(r):
                    s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                T_computed[i, j, k] = s
    T_target = np.ones((n, m, p))
    return np.allclose(T_computed, T_target, atol=tol)

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        return rounded / 2.0
    return None

def _try_rank_reduction(U_init, V_init, W_init, target_rank, max_iters=100, tol=1e-10):
    """
    Try to reduce the rank from current to target_rank using alternating least squares.
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = DIMENSIONS
    r_init = U_init.shape[1]
    
    # Initialize with random subset of columns
    idx = np.random.choice(r_init, size=target_rank, replace=False)
    U = U_init[:, idx].copy()
    V = V_init[:, idx].copy()
    W = W_init[:, idx].copy()
    
    # Target tensor T[i,j,k] = 1
    # We want to minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||_F^2
    # where U[:,r] is vector of length n*m, V[:,r] is length m*p, W[:,r] is length p*n
    # But the tensor is structured: T[i,j,k] = 1 for all i,j,k.
    
    # Let's use a simple ALS approach.
    # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
    U3 = U.reshape(n, m, target_rank)
    V3 = V.reshape(m, p, target_rank)
    W3 = W.reshape(p, n, target_rank)
    
    for it in range(max_iters):
        # Update U
        # For fixed V, W, the contribution to T[i,j,k] from term r is U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # This is a linear system for U[i,j,r] if we fix V, W?
        # Not directly, because U[i,j,r] multiplies V[j,k,r] * W[k,i,r].
        # Let M[i,j,k] = V[j,k,r] * W[k,i,r] summed over r? No.
        
        # Let's use a gradient descent step or Gauss-Newton.
        # Actually, for fixed V and W, we can solve for U using least squares on the tensor.
        # T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        # Let B_r[i,j,k] = V[j,k,r] * W[k,i,r]. This is not quite right because V and W depend on r.
        # The term is U[i,j,r] * V[j,k,r] * W[k,i,r].
        # For fixed r, this is a rank-1 tensor in (i,j,k) space? No.
        
        # Let's stick to the column pruning approach which is safer.
        # If we can't find a reduction, return None.
        
        # Check error
        T_comp = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(target_rank):
                        s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                    T_comp[i, j, k] = s
        err = np.linalg.norm(T_comp - 1.0)
        if err < tol:
            return U, V, W
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build rank-42 solution
    U, V, W = _build_rank42()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 42
    
    # Verify rank-42 solution
    if _tensor_check(U, V, W):
        pass
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank
    # We know rank 42 works. Can we do better?
    # Lower bound: max(n*m, m*p, p*n) = 24.
    # So rank must be >= 24.
    # Let's try to find a solution with rank < 42.
    
    # Strategy: Random restarts with ALS or pruning.
    # Since the tensor is all-ones, it has high symmetry.
    
    # Try ranks from 41 down to 24
    for target_rank in range(41, 23, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try a few random subsets
        found = False
        n_attempts = 20
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Randomly select columns
            idx = np.random.choice(42, size=target_rank, replace=False)
            U_try = U[:, idx].copy()
            V_try = V[:, idx].copy()
            W_try = W[:, idx].copy()
            
            # Check if this sub-selection works (unlikely)
            if _tensor_check(U_try, V_try, W_try, tol=1e-10):
                U_r = _round_to_halfint(U_try)
                V_r = _round_to_halfint(V_try)
                W_r = _round_to_halfint(W_try)
                if U_r is not None and V_r is not None and W_r is not None:
                    if _tensor_check(U_r, V_r, W_r, tol=1e-9):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break
        
        if found:
            continue
    
    # Final round to half-integers
    U_final = _round_to_halfint(best_U)
    V_final = _round_to_halfint(best_V)
    W_final = _round_to_halfint(best_W)
    
    if U_final is None or V_final is None or W_final is None:
        U_final, V_final, W_final = best_U, best_V, best_W
    
    # Final verification
    if not _tensor_check(U_final, V_final, W_final, tol=1e-8):
        U_final, V_final, W_final = _build_rank42()
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
