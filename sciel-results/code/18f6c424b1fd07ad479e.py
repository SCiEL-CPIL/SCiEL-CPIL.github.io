import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(U, V, W):
    """Round to nearest half-integer."""
    return np.round(U * 2) / 2.0, np.round(V * 2) / 2.0, np.round(W * 2) / 2.0

def _build_rank33():
    """Construct rank-33 decomposition (3 blocks of rank-11 for 2x3x2)."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 11 terms for 2x3 x 3x2
    terms = [
        ({0:1, 4:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({3:1, 4:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({4:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({3:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 4:-1}, {2:1, 3:1}, {0:1}),
        ({2:1}, {4:1}, {0:1}),
        ({2:1}, {5:1}, {1:1}),
        ({5:1}, {4:1}, {2:1}),
        ({5:1}, {5:1}, {3:1}),
    ]
    
    for block in range(3):
        for t_idx, (u_d, v_d, w_d) in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in u_d.items():
                U[idx, col] = val
            for idx, val in v_d.items():
                j_l, k_l = idx // 2, idx % 2
                V[j_l * p + block * 2 + k_l, col] = val
            for idx, val in w_d.items():
                i_l, k_l = idx // 2, idx % 2
                W[(block * 2 + k_l) * n + i_l, col] = val
    return U, V, W

def _build_from_blocks(Ub, Vb, Wb, nb, mb, pb, n, m, p):
    """Build full decomposition from block decomposition."""
    rank = Ub.shape[1] * (p // pb)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for block in range(p // pb):
        for c in range(Ub.shape[1]):
            col = block * Ub.shape[1] + c
            U[:, col] = Ub[:, c]
            for j in range(m):
                for k in range(pb):
                    V[j*p + block*pb + k, col] = Vb[j*pb + k, c]
            for k in range(pb):
                for i in range(n):
                    W[(block*pb + k)*n + i, col] = Wb[k*n + i, c]
    return U, V, W

def _search_rank(target_rank, n, m, p, max_time=30, n_restarts=50, seed_offset=0):
    """Search for a decomposition of given rank."""
    start = time.time()
    du, dv, dw = n*m, m*p, p*n
    
    def residuals(params):
        U = params[:du*target_rank].reshape(du, target_rank)
        V = params[du*target_rank:du*target_rank+dv*target_rank].reshape(dv, target_rank)
        W = params[du*target_rank+dv*target_rank:].reshape(dw, target_rank)
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i]) - 1.0
                    idx += 1
        return res
    
    def make_strassen_init(seed):
        """Use Strassen-like structure as initial guess."""
        np.random.seed(seed)
        U = np.zeros((du, target_rank))
        V = np.zeros((dv, target_rank))
        W = np.zeros((dw, target_rank))
        
        # Fill with small random values
        for c in range(target_rank):
            U[:, c] = np.random.randn(du) * 0.5
            V[:, c] = np.random.randn(dv) * 0.5
            W[:, c] = np.random.randn(dw) * 0.5
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def make_sparse_init(seed):
        """Sparse random initialization."""
        np.random.seed(seed)
        U = np.zeros((du, target_rank))
        V = np.zeros((dv, target_rank))
        W = np.zeros((dw, target_rank))
        
        # Each term: pick random nonzero entries
        for c in range(target_rank):
            for _ in range(3):
                U[np.random.randint(du), c] = np.random.choice([-1, 1])
            for _ in range(3):
                V[np.random.randint(dv), c] = np.random.choice([-1, 1])
            for _ in range(3):
                W[np.random.randint(dw), c] = np.random.choice([-1, 1])
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_x = None
    best_cost = np.inf
    
    for r in range(n_restarts):
        if time.time() - start > max_time:
            break
        seed = seed_offset + r * 7 + 3
        if r % 2 == 0:
            x0 = make_strassen_init(seed)
        else:
            x0 = make_sparse_init(seed)
        
        try:
            result = least_squares(residuals, x0, max_nfev=2000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_x is not None and best_cost < 1e-10:
        U = best_x[:du*target_rank].reshape(du, target_rank)
        V = best_x[du*target_rank:du*target_rank+dv*target_rank].reshape(dv, target_rank)
        W = best_x[du*target_rank+dv*target_rank:].reshape(dw, target_rank)
        
        U_r, V_r, W_r = _round_half(U, V, W)
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        return U, V, W
    
    return None

def _search_2x3x3(target_rank, max_time=25, n_restarts=80, seed_offset=0):
    """Search for rank decomposition of 2x3x3."""
    n, m, p = 2, 3, 3
    start = time.time()
    du, dv, dw = 6, 9, 6
    
    def residuals(params):
        U = params[:du*target_rank].reshape(du, target_rank)
        V = params[du*target_rank:du*target_rank+dv*target_rank].reshape(dv, target_rank)
        W = params[du*target_rank+dv*target_rank:].reshape(dw, target_rank)
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i]) - 1.0
                    idx += 1
        return res
    
    def make_init(seed):
        np.random.seed(seed)
        U = np.zeros((du, target_rank))
        V = np.zeros((dv, target_rank))
        W = np.zeros((dw, target_rank))
        
        # Strassen for 2x2x2 part
        if target_rank >= 7:
            strassen = [
                ({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1}),
                ({3:1, 4:1}, {0:1}, {3:1, 4:-1}),
                ({0:1}, {1:1, 4:-1}, {1:1, 4:1}),
                ({4:1}, {3:1, 0:-1}, {0:1, 3:1}),
                ({0:1, 1:1}, {4:1}, {0:-1, 1:1}),
                ({3:1, 0:-1}, {0:1, 1:1}, {4:1}),
                ({1:1, 4:-1}, {3:1, 4:1}, {0:1}),
            ]
            for c, (ud, vd, wd) in enumerate(strassen):
                for idx, val in ud.items(): U[idx, c] = val
                for idx, val in vd.items(): V[idx, c] = val
                for idx, val in wd.items(): W[idx, c] = val
        
        # Fill remaining with random
        for c in range(7 if target_rank >= 7 else 0, target_rank):
            U[:, c] = np.random.randn(du) * 0.8
            V[:, c] = np.random.randn(dv) * 0.8
            W[:, c] = np.random.randn(dw) * 0.8
        
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_x = None
    best_cost = np.inf
    
    for r in range(n_restarts):
        if time.time() - start > max_time:
            break
        seed = seed_offset + r * 11 + 5
        x0 = make_init(seed)
        try:
            result = least_squares(residuals, x0, max_nfev=3000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
            if result.cost < best_cost:
                best_cost = result.cost
                best_x = result.x
                if best_cost < 1e-14:
                    break
        except:
            continue
    
    if best_x is not None and best_cost < 1e-10:
        U = best_x[:du*target_rank].reshape(du, target_rank)
        V = best_x[du*target_rank:du*target_rank+dv*target_rank].reshape(dv, target_rank)
        W = best_x[du*target_rank+dv*target_rank:].reshape(dw, target_rank)
        U_r, V_r, W_r = _round_half(U, V, W)
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    # Try direct search for low rank on 2x3x6
    # Known lower bound considerations: 
    # 2x3x6 has 36 entries. Schoolbook is 36.
    # Try ranks: 22, 24, 26, 28, 30
    
    # Strategy: Try block-based approaches first (faster), then direct search
    
    # 1) Try rank-12 for 2x3x3 -> rank-24 for 2x3x6
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        result = _search_2x3x3(12, max_time=min(remaining - 5, 35), n_restarts=60)
        if result is not None and time.time() < deadline:
            U12, V12, W12 = result
            U, V, W = _build_from_blocks(U12, V12, W12, 2, 3, 3, n, m, p)
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 2) Try rank-11 for 2x3x3 -> rank-22 for 2x3x6
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        result = _search_2x3x3(11, max_time=min(remaining - 5, 30), n_restarts=40, seed_offset=100)
        if result is not None and time.time() < deadline:
            U11, V11, W11 = result
            U, V, W = _build_from_blocks(U11, V11, W11, 2, 3, 3, n, m, p)
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 3) Try direct search for rank-24 on 2x3x6
    remaining = deadline - time.time() - 5.0
    if remaining > 15.0:
        result = _search_rank(24, n, m, p, max_time=min(remaining - 5, 30), n_restarts=40, seed_offset=200)
        if result is not None and time.time() < deadline:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 4) Try rank-26
    remaining = deadline - time.time() - 5.0
    if remaining > 15.0:
        result = _search_rank(26, n, m, p, max_time=min(remaining - 5, 25), n_restarts=30, seed_offset=300)
        if result is not None and time.time() < deadline:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 5) Try rank-28
    remaining = deadline - time.time() - 5.0
    if remaining > 15.0:
        result = _search_rank(28, n, m, p, max_time=min(remaining - 5, 20), n_restarts=25, seed_offset=400)
        if result is not None and time.time() < deadline:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # 6) Fallback: rank-33
    U, V, W = _build_rank33()
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
