<<<<<<< SEARCH
def _try_global_rank_reduction(U, V, W, n, m, p, target_rank, max_iters=50, seed=0):
    """
    Try to reduce the rank of a decomposition from current_rank to target_rank.
    Uses iterative projection.
    """
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return U, V, W
    
    rng = np.random.RandomState(seed)
    
    # Build target tensor
    T = np.zeros((n * m, p * n, p * n), dtype=np.float64)  # placeholder, will fix
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_err = np.inf
    best_UVW = (U.copy(), V.copy(), W.copy())
    
    for attempt in range(max_iters):
        # Start from current solution, project to target_rank
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        for inner_iter in range(100):
            # Solve for W given U, V with target_rank terms
            A_c = np.zeros((n * m, target_rank))
            for t in range(target_rank):
                A_c[:, t] = (U_curr[:, t][:, None] * V_curr[:, t][None, :]).flatten()
            
            W_new = np.zeros((p * n, target_rank))
            for c in range(p * n):
                rhs = T[:, :, c].flatten()
                sol, _, _, _ = np.linalg.lstsq(A_c, rhs, rcond=None)
                W_new[c, :] = sol
            W_curr = W_new
            
            # Solve for U
            A_a = np.zeros((m * p, target_rank))
            for t in range(target_rank):
                A_a[:, t] = (V_curr[:, t][:, None] * W_curr[:, t][None, :]).flatten()
            
            U_new = np.zeros((n * m, target_rank))
            for a in range(n * m):
                rhs = T[a, :, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_a, rhs, rcond=None)
                U_new[a, :] = sol
            U_curr = U_new
            
            # Solve for V
            A_b = np.zeros((n * p, target_rank))
            for t in range(target_rank):
                A_b[:, t] = (U_curr[:, t][:, None] * W_curr[:, t][None, :]).flatten()
            
            V_new = np.zeros((m * p, target_rank))
            for b in range(m * p):
                rhs = T[:, b, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_b, rhs, rcond=None)
                V_new[b, :] = sol
            V_curr = V_new
            
            # Check error
            C = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for t in range(target_rank):
                C += np.einsum('a,b,c->abc', U_curr[:, t], V_curr[:, t], W_curr[:, t])
            err = np.max(np.abs(C - T))
            if err < 1e-10:
                return U_curr, V_curr, W_curr
        
        C = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for t in range(target_rank):
            C += np.einsum('a,b,c->abc', U_curr[:, t], V_curr[:, t], W_curr[:, t])
        err = np.max(np.abs(C - T))
        if err < best_err:
            best_err = err
            best_UVW = (U_curr.copy(), V_curr.copy(), W_curr.copy())
    
    if best_err < 1e-10:
        return best_UVW
    return U, V, W
=======
def _try_global_rank_reduction(U, V, W, n, m, p, target_rank, max_iters=30, seed=0, deadline=None):
    """
    Try to reduce the rank of a decomposition from current_rank to target_rank.
    Uses iterative alternating least squares.
    """
    current_rank = U.shape[1]
    if target_rank >= current_rank:
        return U, V, W
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    best_err = np.inf
    best_UVW = None
    
    for attempt in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Initialize: use first target_rank columns
        U_curr = U[:, :target_rank].copy()
        V_curr = V[:, :target_rank].copy()
        W_curr = W[:, :target_rank].copy()
        
        for inner_iter in range(80):
            if deadline is not None and time.time() > deadline - 3:
                break
            
            # Solve for W given U, V
            A_c = np.zeros((n * m, target_rank))
            for t in range(target_rank):
                A_c[:, t] = (U_curr[:, t][:, None] * V_curr[:, t][None, :]).flatten()
            
            W_new = np.zeros((p * n, target_rank))
            for c in range(p * n):
                rhs = T[:, :, c].flatten()
                sol, _, _, _ = np.linalg.lstsq(A_c, rhs, rcond=None)
                W_new[c, :] = sol
            W_curr = W_new
            
            # Solve for U
            A_a = np.zeros((m * p, target_rank))
            for t in range(target_rank):
                A_a[:, t] = (V_curr[:, t][:, None] * W_curr[:, t][None, :]).flatten()
            
            U_new = np.zeros((n * m, target_rank))
            for a in range(n * m):
                rhs = T[a, :, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_a, rhs, rcond=None)
                U_new[a, :] = sol
            U_curr = U_new
            
            # Solve for V
            A_b = np.zeros((n * p, target_rank))
            for t in range(target_rank):
                A_b[:, t] = (U_curr[:, t][:, None] * W_curr[:, t][None, :]).flatten()
            
            V_new = np.zeros((m * p, target_rank))
            for b in range(m * p):
                rhs = T[:, b, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_b, rhs, rcond=None)
                V_new[b, :] = sol
            V_curr = V_new
            
            # Check error
            C = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for t in range(target_rank):
                C += np.einsum('a,b,c->abc', U_curr[:, t], V_curr[:, t], W_curr[:, t])
            err = np.max(np.abs(C - T))
            if err < 1e-10:
                return U_curr, V_curr, W_curr
        
        C = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for t in range(target_rank):
            C += np.einsum('a,b,c->abc', U_curr[:, t], V_curr[:, t], W_curr[:, t])
        err = np.max(np.abs(C - T))
        if err < best_err:
            best_err = err
            best_UVW = (U_curr.copy(), V_curr.copy(), W_curr.copy())
    
    if best_UVW is not None and best_err < 1e-10:
        return best_UVW
    return U, V, W
>>>>>>> REPLACE

