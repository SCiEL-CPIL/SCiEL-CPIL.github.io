import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
_START = time.time()
DEADLINE = _START + 140.0

def _remaining():
    return DEADLINE - time.time()

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _recon(U, V, W):
    return np.einsum('al,bl,cl->abc', U, V, W)

def _res(U, V, W, T):
    return np.linalg.norm(_recon(U, V, W) - T)

def _als(U, V, W, T, iters=300, tol=1e-13):
    r = U.shape[1]
    eye = np.eye(r)
    prev = _res(U, V, W, T)
    for it in range(iters):
        if _remaining() < 8: break
        # U
        G = (V.T @ V) * (W.T @ W)
        G = G + 1e-10 * (np.trace(G) / r) * eye
        rhs = np.einsum('acl,cl->al', np.einsum('abc,bl->acl', T, V), W)
        U = np.linalg.solve(G, rhs.T).T
        # V
        G = (U.T @ U) * (W.T @ W)
        G = G + 1e-10 * (np.trace(G) / r) * eye
        rhs = np.einsum('abl,al->bl', np.einsum('abc,cl->abl', T, W), U)
        V = np.linalg.solve(G, rhs.T).T
        # W
        G = (U.T @ U) * (V.T @ V)
        G = G + 1e-10 * (np.trace(G) / r) * eye
        rhs = np.einsum('bcl,bl->cl', np.einsum('abc,al->bcl', T, U), V)
        W = np.linalg.solve(G, rhs.T).T
        cur = _res(U, V, W, T)
        if abs(prev - cur) < tol:
            prev = cur
            break
        prev = cur
    return U, V, W, cur

def _snap_rational(U, V, W, T, max_denom=8):
    """Try to snap to rationals with small denominators"""
    for denom in [1, 2, 4, max_denom]:
        factor = float(denom)
        Us = np.round(U * factor) / factor
        Vs = np.round(V * factor) / factor
        Ws = np.round(W * factor) / factor
        if _res(Us, Vs, Ws, T) < 1e-6:
            return Us, Vs, Ws, True
    return None, None, None, False

def _polish(U, V, W, T, iters=100):
    """High-precision polish pass"""
    r = U.shape[1]
    eye = np.eye(r)
    for it in range(iters):
        if _remaining() < 5: break
        # U
        G = (V.T @ V) * (W.T @ W)
        G = G + 1e-14 * (np.trace(G) / r) * eye
        rhs = np.einsum('acl,cl->al', np.einsum('abc,bl->acl', T, V), W)
        U_new = np.linalg.solve(G, rhs.T).T
        # V
        G = (U_new.T @ U_new) * (W.T @ W)
        G = G + 1e-14 * (np.trace(G) / r) * eye
        rhs = np.einsum('abl,al->bl', np.einsum('abc,cl->abl', T, W), U_new)
        V_new = np.linalg.solve(G, rhs.T).T
        # W
        G = (U_new.T @ U_new) * (V_new.T @ V_new)
        G = G + 1e-14 * (np.trace(G) / r) * eye
        rhs = np.einsum('bcl,bl->cl', np.einsum('abc,al->bcl', T, U_new), V_new)
        W_new = np.linalg.solve(G, rhs.T).T
        U, V, W = U_new, V_new, W_new
        cur = _res(U, V, W, T)
        if cur < 1e-15:
            break
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
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

def _structured_init(n, m, p, r, rng, seed_offset=0):
    """Create structured initializations that mix random and pattern-based approaches"""
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # Mix of random and structured
    for col in range(r):
        if col % 3 == 0:
            # Random
            U[:, col] = rng.standard_normal(n*m)
            V[:, col] = rng.standard_normal(m*p)
            W[:, col] = rng.standard_normal(p*n)
        elif col % 3 == 1:
            # Sparse random
            mask_u = rng.random(n*m) < 0.5
            mask_v = rng.random(m*p) < 0.5
            mask_w = rng.random(p*n) < 0.5
            U[:, col] = np.where(mask_u, rng.standard_normal(n*m), 0)
            V[:, col] = np.where(mask_v, rng.standard_normal(m*p), 0)
            W[:, col] = np.where(mask_w, rng.standard_normal(p*n), 0)
        else:
            # Pattern-based: each column focuses on a subset of indices
            i_idx = rng.integers(0, n)
            j_idx = rng.integers(0, m)
            k_idx = rng.integers(0, p)
            U[i_idx*m + j_idx, col] = 1.0
            V[j_idx*p + k_idx, col] = 1.0
            W[k_idx*n + i_idx, col] = 1.0
    
    # Normalize to prevent numerical issues
    for col in range(r):
        nu = np.linalg.norm(U[:, col])
        nv = np.linalg.norm(V[:, col])
        nw = np.linalg.norm(W[:, col])
        if nu > 0 and nv > 0 and nw > 0:
            scale = (nu * nv * nw) ** (1.0/3.0)
            U[:, col] /= scale
            V[:, col] /= scale
            W[:, col] /= scale
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    best = None  # (rank, res, U, V, W)
    
    # Candidate ranks - start from known lower bounds
    # For 3x4x5, schoolbook is 60. Known lower bounds suggest < 60 is possible.
    ranks = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]
    rng = np.random.default_rng(42)
    
    for r in ranks:
        if _remaining() < 15: break
        # More restarts for smaller ranks
        nrestart = max(5, min(30, int(4000 / r)))
        found = False
        
        for it in range(nrestart):
            if _remaining() < 10: break
            
            # Use structured initialization
            U, V, W = _structured_init(n, m, p, r, rng, seed_offset=it)
            
            # Run ALS
            U, V, W, res = _als(U, V, W, T, iters=500, tol=1e-14)
            
            if res < 1e-8:
                # Polish with higher precision
                U, V, W = _polish(U, V, W, T, iters=200)
                res = _res(U, V, W, T)
                
                if res < 1e-7:
                    found = True
                    if best is None or r < best[0]:
                        best = (r, res, U, V, W)
                    break  # Found at this rank, move to next (lower) rank
        
        if found:
            # We found a solution at rank r; since we go ascending, this is the min found so far
            # Continue to check if a lower rank also works (we already tried lower in this iteration)
            pass
    
    # Pick best
    if best is not None:
        r, res, U, V, W = best
        # Try to snap to rationals
        Us, Vs, Ws, ok = _snap_rational(U, V, W, T, max_denom=8)
        if ok:
            U, V, W = Us, Vs, Ws
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _schoolbook(n, m, p)
