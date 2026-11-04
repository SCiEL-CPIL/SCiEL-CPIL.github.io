import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _matmul_check(U, V, W, n, m, p):
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

def _als_decompose(n, m, p, r, n_restarts=5, n_iters=50, seed=42):
    """Try to find a rank-r decomposition using ALS."""
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_sol = None
    
    for restart in range(n_restarts):
        # Random initialization
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        for it in range(n_iters):
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    b = np.ones(p)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = x
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    b = np.ones(n)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = x
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    b = np.ones(m)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = x
        
        # Check error
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
        err = np.max(np.abs(T - 1.0))
        
        if err < best_err:
            best_err = err
            best_sol = (U.copy(), V.copy(), W.copy())
    
    return best_sol, best_err

def _round_to_half(U, V, W):
    """Round entries to nearest 0.5."""
    U2 = np.round(2*U)/2
    V2 = np.round(2*V)/2
    W2 = np.round(2*W)/2
    return U2, V2, W2

def _construct_rank56():
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

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110  # Leave 10s margin
    
    # Try to find a lower-rank decomposition
    best = None
    
    for r in [52, 50, 48, 44, 40, 36, 32]:
        if time.time() > deadline:
            break
        try:
            sol, err = _als_decompose(n, m, p, r, n_restarts=3, n_iters=30, seed=r)
            if sol is not None and err < 0.1:  # Preliminary check
                U, V, W = sol
                U2, V2, W2 = _round_to_half(U, V, W)
                U_list = U2.tolist()
                V_list = V2.tolist()
                W_list = W2.tolist()
                ok, err2 = _matmul_check(U_list, V_list, W_list, n, m, p)
                if ok:
                    U_t, V_t, W_t, r_t = _trim_zero_columns(U_list, V_list, W_list)
                    best = (r_t, U_t, V_t, W_t)
                    break  # Found a valid lower-rank solution
        except Exception:
            pass
    
    # Fallback to rank-56
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            if best is None or r < best[0]:
                best = (r, U, V, W)
    except Exception:
        pass
    
    if best is None:
        cand = _construct_rank56()
        return cand
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
