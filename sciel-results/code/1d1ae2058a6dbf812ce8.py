import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Strassen coefficients for 2x2 matrix multiplication
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]

    # Build rank-21 fallback
    rank_fb = 21
    U_fb = [[0] * rank_fb for _ in range(4)]
    V_fb = [[0] * rank_fb for _ in range(12)]
    W_fb = [[0] * rank_fb for _ in range(12)]

    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U_fb[idx][col] = sc['U'][idx]
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V_fb[v_idx[idx]][col] = sc['V'][idx]
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W_fb[w_idx[idx]][col] = sc['W'][idx]

    # Quick verification of fallback
    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        S = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        T = np.zeros((4, 12, 12))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    T[a, b, c] = 1.0
        return np.max(np.abs(S - T)) < 1e-8

    if not verify(U_fb, V_fb, W_fb):
        # Should never happen, but just in case
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    # Try to find rank-20 solution using a structured approach
    # Key insight: The A matrix is shared, so we might be able to combine
    # some Strassen steps across blocks.
    
    # Approach: Try a parametric family where we allow cross-block coupling
    # in the U factors while keeping V and W block-diagonal.
    
    # For rank 20, we need to save 1 rank from 21.
    # Try: Can we express one of the 21 rank-1 terms as a combination
    # of others with slightly modified V/W factors?
    
    # Alternative: Use the fact that for the 2x2 A matrix, the 7 Strassen
    # products have linear dependencies when viewed across multiple B blocks.
    
    # Let's try a direct optimization with a better initialization.
    # Use the rank-21 solution and try to null out one dimension.
    
    import numpy as np
    
    T = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T[a, b, c] = 1.0

    target_rank = 20
    dim_U, dim_V, dim_W = 4, 12, 12
    total_params = target_rank * (dim_U + dim_V + dim_W)

    # Build rank-21 as numpy
    U21 = np.array(U_fb, dtype=float)  # (4, 21)
    V21 = np.array(V_fb, dtype=float)  # (12, 21)
    W21 = np.array(W_fb, dtype=float)  # (12, 21)

    # Strategy: Try to find a rank-20 by looking for a null vector in the
    # 21-dimensional space of factorizations.
    # 
    # A tensor T has rank r if there exist U, V, W of size (., r) such that
    # T = sum_s U_s ⊗ V_s ⊗ W_s.
    #
    # If T has a rank-21 decomposition, the set of all rank-21 decompositions
    # forms a variety. If the rank is actually 20, this variety has a
    # 1-dimensional component (or more) that we can find.
    #
    # Practical approach: Take the 21 columns, and look for a linear
    # combination that gives zero when plugged into the tensor equation.
    # i.e., find (α_s) not all zero such that sum_s α_s (U_s ⊗ V_s ⊗ W_s) = 0
    # If such a dependency exists, we can eliminate one column.
    
    # Check for linear dependencies in the rank-21 decomposition
    # by flattening the 21 rank-1 tensors and checking rank.
    flat_tensors = []
    for s in range(21):
        U_s = U21[:, s]
        V_s = V21[:, s]
        W_s = W21[:, s]
        tensor_s = np.einsum('a,b,c->abc', U_s, V_s, W_s)
        flat_tensors.append(tensor_s.flatten())
    
    flat_array = np.array(flat_tensors)  # (21, 576)
    
    # Check the rank of this 21x576 matrix
    # If rank < 21, there's a dependency
    rank_check = np.linalg.matrix_rank(flat_array, tol=1e-10)
    
    if rank_check < 21:
        # There's a linear dependency! Find it and use it to reduce rank.
        # Find null vector
        _, S, Vt = np.linalg.svd(flat_array, full_matrices=True)
        # The last row of Vt corresponds to the smallest singular value
        if S[-1] < 1e-8:
            null_vec = Vt[-1]
            # Check if this null vector allows rank-20
            # The null vector tells us sum_s null_vec[s] * T_s = 0
            # This means we can express one tensor as a combination of others
            # but this doesn't directly give a rank-20 decomposition of T.
            # 
            # Actually, a linear dependency among the rank-1 terms means
            # the decomposition is not unique, but doesn't necessarily mean
            # rank is lower. We need to check if T itself has rank 20.
            pass

    # More direct approach: Try optimization with multiple random restarts
    # but with a time-limited budget.
    
    from scipy.optimize import least_squares
    
    def residuals(params):
        U_s = params[0:target_rank * dim_U].reshape(target_rank, dim_U)
        V_s = params[target_rank * dim_U:target_rank * (dim_U + dim_V)].reshape(target_rank, dim_V)
        W_s = params[target_rank * (dim_U + dim_V):].reshape(target_rank, dim_W)
        S = np.einsum('sa,sb,sc->abc', U_s, V_s, W_s)
        return (S - T).flatten()

    found = False
    best_params = None
    best_cost = np.inf
    
    np.random.seed(12345)
    
    # Try a few random initializations with limited iterations
    max_trials = 5
    for trial in range(max_trials):
        if time.time() > deadline - 20:
            break
        
        init = np.random.randn(total_params) * 0.5
        try:
            result = least_squares(residuals, init, method='lm', max_nfev=1000, 
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
            if cost < 1e-10:
                found = True
                break
        except Exception:
            continue

    # Also try initializing from the rank-21 solution with one column removed
    for remove_col in range(min(21, 5)):
        if time.time() > deadline - 15:
            break
        
        cols_to_keep = [c for c in range(21) if c != remove_col]
        init = np.zeros(total_params)
        for new_col, old_col in enumerate(cols_to_keep):
            init[new_col * dim_U:(new_col + 1) * dim_U] = U21[:, old_col]
            init[(new_col + 1) * dim_U:(new_col + 2) * dim_U] = V21[:, old_col]
            init[(new_col + 2) * dim_U:(new_col + 3) * dim_U] = W21[:, old_col]
        
        init += np.random.randn(total_params) * 0.01
        
        try:
            result = least_squares(residuals, init, method='lm', max_nfev=1500,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
            if cost < 1e-10:
                found = True
                break
        except Exception:
            continue

    if found and best_params is not None:
        # Verify the solution
        U_s = best_params[0:target_rank * dim_U].reshape(target_rank, dim_U)
        V_s = best_params[target_rank * dim_U:target_rank * (dim_U + dim_V)].reshape(target_rank, dim_V)
        W_s = best_params[target_rank * (dim_U + dim_V):].reshape(target_rank, dim_W)
        S = np.einsum('sa,sb,sc->abc', U_s, V_s, W_s)
        if np.max(np.abs(S - T)) < 1e-6:
            U_out = [[0] * target_rank for _ in range(dim_U)]
            V_out = [[0] * target_rank for _ in range(dim_V)]
            W_out = [[0] * target_rank for _ in range(dim_W)]
            for s in range(target_rank):
                for idx in range(dim_U):
                    U_out[idx][s] = U_s[s, idx]
                for idx in range(dim_V):
                    V_out[idx][s] = V_s[s, idx]
                for idx in range(dim_W):
                    W_out[idx][s] = W_s[s, idx]
            return {"U": U_out, "V": V_out, "W": W_out}

    # Return the rank-21 fallback
    return {"U": U_fb, "V": V_fb, "W": W_fb}
