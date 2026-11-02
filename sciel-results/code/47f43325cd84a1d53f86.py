def _try_rank_10_search(deadline):
    """
    Attempt to find a rank 10 decomposition using numerical optimization.
    Optimized for speed to ensure we return within time limits.
    """
    n, m, p = DIMENSIONS
    rank = 10
    T_target = _get_target_tensor()
    
    best_error = float('inf')
    best_solution = None
    
    # Reduced budget: We know rank 11 is likely minimal.
    # Spend at most 5 seconds on this search.
    max_time = 5.0
    
    num_restarts = 10
    steps_per_restart = 20
    learning_rate = 0.05
    
    for restart in range(num_restarts):
        if time.time() > deadline - max_time:
            break
            
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for step in range(steps_per_restart):
            if time.time() > deadline - max_time:
                break
                
            T_calc = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T_calc - T_target)
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
                
            if error < 1e-8:
                return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
                
            dT = T_calc - T_target
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # Vectorized gradient calculation
            # grad_U[i, r] = sum_{j,k} dT[i, j, k] * V[j, r] * W[k, r]
            # Reshape dT to (n*m, m*p*p*n)? No.
            # Use einsum for gradients too
            # grad_U = np.einsum('ijk,jr,kr->ir', dT, V, W)
            # grad_V = np.einsum('ijk,ir,kr->jr', dT, U, W)
            # grad_W = np.einsum('ijk,ir,jr->kr', dT, U, V)
            
            grad_U = np.einsum('ijk,jr,kr->ir', dT, V, W)
            grad_V = np.einsum('ijk,ir,kr->jr', dT, U, W)
            grad_W = np.einsum('ijk,ir,jr->kr', dT, U, V)
            
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            W -= learning_rate * grad_W
            
            U /= np.linalg.norm(U) + 1e-9
            V /= np.linalg.norm(V) + 1e-9
            W /= np.linalg.norm(W) + 1e-9

    if best_solution is not None:
        U, V, W = best_solution
        if _verify(U, V, W, T_target):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None
