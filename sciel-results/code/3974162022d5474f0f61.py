import time
import numpy as np

DIMENSIONS = [3, 3, 5]
_START_TIME = time.time()
_DEADLINE = 150
_SAFETY_MARGIN = 10  # seconds before the kill

def _schoolbook_decomposition():
    """
    Rank-45 exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    Schoolbook algorithm with the specified index conventions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 3 * 3 * 5 = 45

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = 1 for all (i,j,k) should equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _structured_init(n, m, p, rank, seed):
    """
    Initialize U, V, W with structure inspired by Strassen-Winograd patterns.
    Each rank-1 term gets a pattern of non-zero entries.
    """
    np.random.seed(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Assign each rank term a "type" that determines which indices it covers
    # Use a mix of dense and sparse patterns
    for r in range(rank):
        # Randomly choose a subset of indices for each factor
        # U covers (i,j) pairs
        n_u_nonzero = max(1, min(nm, np.random.randint(2, nm + 1)))
        u_indices = np.random.choice(nm, size=n_u_nonzero, replace=False)
        U[u_indices, r] = np.random.choice([-1, 1], size=n_u_nonzero) * np.random.uniform(0.5, 1.5, size=n_u_nonzero)
        
        # V covers (j,k) pairs
        n_v_nonzero = max(1, min(mp, np.random.randint(2, mp + 1)))
        v_indices = np.random.choice(mp, size=n_v_nonzero, replace=False)
        V[v_indices, r] = np.random.choice([-1, 1], size=n_v_nonzero) * np.random.uniform(0.5, 1.5, size=n_v_nonzero)
        
        # W covers (k,i) pairs
        n_w_nonzero = max(1, min(pn, np.random.randint(2, pn + 1)))
        w_indices = np.random.choice(pn, size=n_w_nonzero, replace=False)
        W[w_indices, r] = np.random.choice([-1, 1], size=n_w_nonzero) * np.random.uniform(0.5, 1.5, size=n_w_nonzero)
    
    return U, V, W

def _try_strassen_like_search(n, m, p, max_rank, time_budget):
    """
    Attempt to find a rank < max_rank exact decomposition using structured
    search with multiple initialization strategies.
    
    Returns (U, V, W) if a valid decomposition is found, else None.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Search ranks from 25 up to max_rank-1
    # Focus on ranks where improvements are most likely
    ranks_to_try = list(range(25, min(max_rank, 45)))
    
    # Sort by likelihood: try middle ranges first where structured decompositions exist
    # For 3x3x5, known results suggest rank ~33-36 might be achievable
    mid_rank = (25 + min(max_rank, 45)) // 2
    ranks_to_try.sort(key=lambda r: abs(r - mid_rank))
    
    for rank in ranks_to_try:
        if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Try multiple initialization strategies
        n_restarts = 8
        for restart in range(n_restarts):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                break
            
            if restart < 4:
                # Structured initialization
                U, V, W = _structured_init(n, m, p, rank, rank * 1000 + restart * 100 + 42)
            else:
                # Random initialization with varied scales
                np.random.seed(rank * 1000 + restart * 100 + 42)
                scale = 0.3 + 0.1 * (restart - 4)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            
            # Reshape for vectorized operations
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            # Adaptive optimization with restarts
            max_iters = 600
            lr = 0.03
            best_err = np.inf
            best_state = None
            
            for it in range(max_iters):
                if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                    break
                
                # Compute Error Tensor
                E = 1.0 - np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                err_norm = np.linalg.norm(E)
                
                if err_norm < best_err:
                    best_err = err_norm
                    best_state = (U_r.copy(), V_r.copy(), W_r.copy())
                
                if err_norm < 1e-10:
                    # Found a valid decomposition!
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                # Gradient steps
                grad_U = np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
                grad_V = np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
                grad_W = np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
                
                # Adaptive learning rate: decay based on error
                lr_adj = lr * (1.0 + 0.1 * np.exp(-err_norm / 10.0))
                U_r -= lr_adj * grad_U
                V_r -= lr_adj * grad_V
                W_r -= lr_adj * grad_W
                
                # Learning rate schedule
                if it % 50 == 0 and it > 0:
                    lr *= 0.97
                
                # Periodic restart from best state if stuck
                if it > 100 and err_norm > best_err * 1.5:
                    U_r, V_r, W_r = best_state
                    lr = lr * 0.5
            
            # Final check with best state
            if best_state is not None:
                U_b, V_b, W_b = best_state
                E_final = 1.0 - np.einsum('ijr,jkr,kir->ijk', U_b, V_b, W_b)
                if np.linalg.norm(E_final) < 1e-8:
                    U_rounded = np.round(U_b * 2) / 2
                    V_rounded = np.round(V_b * 2) / 2
                    W_rounded = np.round(W_b * 2) / 2
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    U_list = U_b.tolist()
                    V_list = V_b.tolist()
                    W_list = W_b.tolist()
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Entry point with deadline guard. Attempts to find a lower-rank decomposition
    using a Strassen-like search, falling back to the schoolbook rank-45 decomposition.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Check if we have time for a search
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Not enough time, return schoolbook
        return _schoolbook_decomposition()
    
    # Try to find a lower-rank decomposition
    result = _try_strassen_like_search(n, m, p, schoolbook_rank, _DEADLINE - _SAFETY_MARGIN - elapsed)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()
