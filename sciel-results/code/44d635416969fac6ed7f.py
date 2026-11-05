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

def _als_decompose(n, m, p, r, n_restarts=5, n_iters=50, seed=42, time_limit=10.0):
    """Try to find a rank-r decomposition using ALS."""
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_sol = None
    t_start = time.time()
    
    for restart in range(n_restarts):
        if time.time() - t_start > time_limit:
            break
            
        # Random initialization
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        for it in range(n_iters):
            # Solve for U
            for i in range(n):
                for j in range(m):
                    # A is p x r, b is p
                    # U[i*m+j] * (V[j*p+k] * W[k*n+i]) = 1 for all k
                    # Let u be the vector U[i*m+j].
                    # Sum_k V[j*p+k, t] W[k*n+i, t] u_t = 1
                    # This is a linear system in u_t.
                    # Matrix M where M[t, t'] = Sum_k V[j*p+k, t] W[k*n+i, t']
                    # We want M u = 1
                    # Actually, the term is Sum_t U[i,m,j,t] V[j,p,k,t] W[k,n,i,t]
                    # For fixed i,j, we want Sum_k (Sum_t U[i,m,j,t] V[j,p,k,t]) W[k,n,i,t] = 1
                    # Let X_k = Sum_t U[i,m,j,t] V[j,p,k,t]
                    # Then Sum_k X_k W[k,n,i,t] = 1.
                    # This is not a simple linear system in U directly if we view it as X.
                    # But X is linear in U.
                    # X_k = <U_i_j, V_j_k>
                    # Sum_k <U_i_j, V_j_k> W_k_i = 1
                    # <U_i_j, Sum_k V_j_k W_k_i> = 1
                    # So U_i_j dot (Sum_k V_j_k W_k_i) = 1.
                    # This is ONE equation for r unknowns?
                    # No, the equation must hold for the scalar result.
                    # The tensor entry T[i,j,k] = Sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                    # We want T[i,j,k] = 1 for all i,j,k.
                    # For fixed i,j, we have p equations (one for each k).
                    # T[i,j,k] = Sum_r U[i,j,r] (V[j,k,r] W[k,i,r]) = 1.
                    # Let A[i,j] be the vector U[i*m+j].
                    # Let b_k be the vector with components b_k[r] = V[j,k,r] W[k,i,r].
                    # Then Sum_k A[i,j] dot b_k ... No.
                    # T[i,j,k] = A[i,j] dot b_k.
                    # We want A[i,j] dot b_k = 1 for all k.
                    # This is a system of p equations: B A[i,j]^T = 1.
                    # Where B is a p x r matrix with B[k, r] = V[j,k,r] W[k,i,r].
                    
                    B = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                    b = np.ones(p)
                    x, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    U[i*m+j, :] = x
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = U[i,m,j] dot (V[j,p,k] W[k,n,i]) = 1
                    # For fixed j,k, we have n equations (one for each i).
                    # Let V_j_k be the vector.
                    # U_i_j dot (V_j_k W_k_i) = 1
                    # (U_i_j W_k_i) dot V_j_k = 1
                    # Let C[i, r] = U[i*m+j, r] W[k*n+i, r]
                    # C V_j_k^T = 1
                    C = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                    b = np.ones(n)
                    x, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                    V[j*p+k, :] = x
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = (U_i_j V_j_k) dot W_k_i = 1
                    # For fixed k,i, we have m equations (one for each j).
                    # Let D[j, r] = U[i*m+j, r] V[j*p+k, r]
                    # D W_k_i^T = 1
                    D = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                    b = np.ones(m)
                    x, _, _, _ = np.linalg.lstsq(D, b, rcond=None)
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
    deadline = t0 + 100  # Leave 20s margin for safety
    
    # Fallback to rank-56
    best = None
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            best = (r, U, V, W)
    except Exception:
        pass
    
    # Try to find a lower-rank decomposition
    # We try ranks lower than the current best
    if best is not None:
        current_rank = best[0]
        # Try a few candidate ranks
        candidate_ranks = [48, 50, 52]
        # Filter out ranks >= current_rank
        candidate_ranks = [r for r in candidate_ranks if r < current_rank]
        
        for r in candidate_ranks:
            if time.time() > deadline:
                break
            try:
                # Use more restarts but fewer iterations to explore landscape
                sol, err = _als_decompose(n, m, p, r, n_restarts=10, n_iters=20, seed=r, time_limit=15.0)
                if sol is not None and err < 0.1:
                    U, V, W = sol
                    # Try rounding to half-integers
                    U2, V2, W2 = _round_to_half(U, V, W)
                    U_list = U2.tolist()
                    V_list = V2.tolist()
                    W_list = W2.tolist()
                    ok, err2 = _matmul_check(U_list, V_list, W_list, n, m, p)
                    if ok:
                        U_t, V_t, W_t, r_t = _trim_zero_columns(U_list, V_list, W_list)
                        if r_t < current_rank:
                            best = (r_t, U_t, V_t, W_t)
                            current_rank = r_t
                            # If we found a very good solution, we might stop
                            if r_t <= 48:
                                break
            except Exception:
                pass
    
    if best is None:
        cand = _construct_rank56()
        return cand
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
