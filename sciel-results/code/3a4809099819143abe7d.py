import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W (they contribute nothing)."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _nullify_columns(U, V, W, target_cols, n, m, p, max_iter=500, lr=0.01, deadline=None):
    """
    Try to nullify specified columns while preserving the tensor decomposition.
    Uses gradient descent on the combined objective:
      minimize: sum over target cols of ||U_col||^2 + ||V_col||^2 + ||W_col||^2
      subject to: tensor equation holds (penalty)
    Returns (U, V, W, success)
    """
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    r = Ua.shape[1]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute tensor
        Ur = Ua.reshape(n, m, r)
        Vr = Va.reshape(m, p, r)
        Wr = Wa.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
        T_err = T - 1.0
        penalty = np.sum(T_err ** 2)
        
        if penalty < 1e-10:
            # Check if target columns are near zero
            target_norm = 0.0
            for c in target_cols:
                target_norm += np.sum(Ua[:, c] ** 2) + np.sum(Va[:, c] ** 2) + np.sum(Wa[:, c] ** 2)
            if target_norm < 1e-8:
                return Ua, Va, Wa, True
        
        # Compute gradients
        # d/dUa of penalty: for each (i,j,r), grad = 2 * sum_{k} T_err[i,j,k] * Vr[j,k,r] * Wr[k,i,r]
        # This is complex; use finite differences for simplicity given small problem size
        
        # Actually, let's use analytical gradients via einsum
        # d(penalty)/dUa[i*m+j, r] = 2 * sum_k T_err[i,j,k] * Vr[j,k,r] * Wr[k,i,r]
        # d(penalty)/dVa[j*p+k, r] = 2 * sum_i T_err[i,j,k] * Ur[i,j,r] * Wr[k,i,r]
        # d(penalty)/dWa[k*n+i, r] = 2 * sum_j T_err[i,j,k] * Ur[i,j,r] * Vr[j,k,r]
        
        # For nullity: d/dUa[i*m+j, r] of ||Ua[:,r]||^2 = 2*Ua[i*m+j,r] (only for target cols)
        
        dUa = np.zeros_like(Ua)
        dVa = np.zeros_like(Va)
        dWa = np.zeros_like(Wa)
        
        # Gradient of penalty w.r.t. U
        # dUa[i*m+j, r] = 2 * sum_k T_err[i,j,k] * Vr[j,k,r] * Wr[k,i,r]
        # Reshape for computation
        T_err_ijk = T_err  # shape (n, m, p)
        Vr_jkr = Vr  # shape (m, p, r)
        Wr_kin = Wr  # shape (p, n, r)
        
        # dUa_ijk_r = 2 * T_err_ijk * Vr_jkr * Wr_kin summed over k
        # dUa[i,j,r] = 2 * sum_k T_err[i,j,k] * Vr[j,k,r] * Wr[k,i,r]
        dUa_ijk = 2.0 * np.einsum('ijk,jkr,kin->ijr', T_err, Vr, Wr)
        dUa = dUa_ijk.reshape(n * m, r)
        
        dVa_jkr = 2.0 * np.einsum('ijk,ijr,kin->jkr', T_err, Ur, Wr)
        dVa = dVa_jkr.reshape(m * p, r)
        
        dWa_kin = 2.0 * np.einsum('ijk,ijr,jkr->kin', T_err, Ur, Vr)
        dWa = dWa_kin.reshape(p * n, r)
        
        # Add nullity gradient for target columns
        for c in target_cols:
            dUa[:, c] += 2.0 * Ua[:, c]
            dVa[:, c] += 2.0 * Va[:, c]
            dWa[:, c] += 2.0 * Wa[:, c]
        
        # Update
        Ua -= lr * dUa
        Va -= lr * dVa
        Wa -= lr * dWa
        
        # Periodically check
        if it % 50 == 0:
            Ur2 = Ua.reshape(n, m, r)
            Vr2 = Va.reshape(m, p, r)
            Wr2 = Wa.reshape(p, n, r)
            T2 = np.einsum('ijr,jkr,kin->ijk', Ur2, Vr2, Wr2)
            pen2 = np.sum((T2 - 1.0) ** 2)
            if pen2 > 1.0:
                # Diverged, revert
                Ua = np.array(U, dtype=float).copy()
                Va = np.array(V, dtype=float).copy()
                Wa = np.array(W, dtype=float).copy()
                return Ua, Va, Wa, False
    
    # Final check
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    penalty = np.sum((T - 1.0) ** 2)
    if penalty < 1e-8:
        target_norm = 0.0
        for c in target_cols:
            target_norm += np.sum(Ua[:, c] ** 2) + np.sum(Va[:, c] ** 2) + np.sum(Wa[:, c] ** 2)
        if target_norm < 1e-6:
            return Ua, Va, Wa, True
    return Ua, Va, Wa, False


def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by nullifying columns from the Strassen solution.
    Returns (U, V, W, new_rank) or None if no improvement.
    """
    r = len(U[0])
    best_rank = r
    best_U, best_V, best_W = U, V, W
    
    # Try nullifying one column at a time, starting from the end
    for target_col in range(r - 1, -1, -1):
        if time.time() > deadline:
            break
        Ua, Va, Wa, success = _nullify_columns(
            best_U, best_V, best_W, [target_col], n, m, p,
            max_iter=200, lr=0.005, deadline=deadline
        )
        if success:
            # Check if we can also nullify more columns
            # For now, just record this improvement
            new_U = [[Ua[i, c] for c in range(r) if c != target_col] for i in range(len(best_U))]
            new_V = [[Va[i, c] for c in range(r) if c != target_col] for i in range(len(best_V))]
            new_W = [[Wa[i, c] for c in range(r) if c != target_col] for i in range(len(best_W))]
            best_U, best_V, best_W = new_U, new_V, new_W
            best_rank = r - 1
            r = r - 1
            # Continue trying to nullify more
    
    if best_rank < r:
        return best_U, best_V, best_W, best_rank
    return None


def _random_search_rank(target_rank, n, m, p, n_trials=1000, deadline=None):
    """
    Randomized search for a decomposition of given rank with entries in {-1, 0, 1}.
    Uses the structure that each column has sparse support.
    """
    rng = np.random.RandomState(42)
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random sparse factors
        U = np.zeros((n_u, target_rank))
        V = np.zeros((n_v, target_rank))
        W = np.zeros((n_w, target_rank))
        
        for c in range(target_rank):
            # Random sparse support
            nz_u = rng.randint(1, 4)
            nz_v = rng.randint(1, 4)
            nz_w = rng.randint(1, 4)
            
            idx_u = rng.choice(n_u, size=nz_u, replace=False)
            idx_v = rng.choice(n_v, size=nz_v, replace=False)
            idx_w = rng.choice(n_w, size=nz_w, replace=False)
            
            for i in idx_u:
                U[i, c] = rng.choice([-1, 1])
            for i in idx_v:
                V[i, c] = rng.choice([-1, 1])
            for i in idx_w:
                W[i, c] = rng.choice([-1, 1])
        
        # Check
        Ur = U.reshape(n, m, target_rank)
        Vr = V.reshape(m, p, target_rank)
        Wr = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
        if np.max(np.abs(T - 1.0)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130  # Safety margin

    best = None
    
    # Start with known rank-56 solution
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            best = (r, U, V, W)
    except Exception:
        pass
    
    if best is None:
        return _construct_rank56()
    
    # Try to reduce rank via column nullification
    try:
        result = _try_rank_reduction(best[1], best[2], best[3], n, m, p, deadline)
        if result is not None:
            U, V, W, r = result
            ok, err = _matmul_check(U, V, W, n, m, p)
            if ok:
                U2, V2, W2, r2 = _trim_zero_columns(U, V, W)
                if r2 < best[0]:
                    best = (r2, U2, V2, W2)
    except Exception:
        pass
    
    # Try random search for lower ranks
    for target_rank in range(32, best[0]):
        if time.time() > deadline:
            break
        try:
            result = _random_search_rank(target_rank, n, m, p, n_trials=200, deadline=deadline)
            if result is not None:
                U, V, W = result
                ok, err = _matmul_check(U, V, W, n, m, p)
                if ok:
                    U2, V2, W2, r2 = _trim_zero_columns(U, V, W)
                    if r2 < best[0]:
                        best = (r2, U2, V2, W2)
                        break  # Found a valid lower rank
        except Exception:
            pass
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
