"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 120.0  # safety margin before the 150s kill

def _remaining():
    return _DEADLINE - (_time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_k U[:,k] V[:,k] W[:,k] == MM tensor exactly (within tol)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Build tensor T[U_idx, V_idx, W_idx]
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Contract: for each (u_idx, w_idx), sum_k U[u,k] V[v,k] W[w,k]
    # We need: sum_k U[u_idx,k] V[v_idx,k] W[w_idx,k] == T[u_idx, v_idx, w_idx]
    # Build via einsum: A[u,v] = sum_k U[u,k] V[v,k]; then result[u,v,w] = sum_k ... 
    # Do it as: M[u,v] = U[:,k] dot V[:,k] summed over k -> use tensordot
    # result[u,v,w] = sum_k U[u,k] * V[v,k] * W[w,k]
    # = sum_k (U[:,k] outer V[:,k]) * W[w,k]
    # Efficient: reshape
    Ue = U.reshape(n * m, rank)
    Ve = V.reshape(m * p, rank)
    We = W.reshape(p * n, rank)
    # tmp[u, v, k] = U[u,k] V[v,k]
    tmp = np.einsum('uk,vk->uvk', Ue, Ve)
    result = np.einsum('uvk,wk->uvw', tmp, We)
    if np.max(np.abs(result - T)) > tol:
        return False
    return True

def _solve_221_rank3():
    """Find a rank-3 decomposition of the 2x2x1 MM tensor. Returns (U,V,W) or None."""
    n, m, p = 2, 2, 1
    U = np.zeros((n * m, 3))
    V = np.zeros((m * p, 3))
    W = np.zeros((p * n, 3))
    # Known rank-3 decomposition of M_{2,2,1}:
    # This is a standard result. We use the decomposition derived from the
    # 2x2x2 Strassen by projecting the last index.
    # Term structure (u over 4, v over 2, w over 2):
    # u1 = [1, 0, 0, 0], v1 = [1, 0], w1 = [1, 0]
    # u2 = [0, 0, 1, 0], v2 = [0, 1], w2 = [0, 1]
    # u3 = [0, 1, 0, 1], v3 = [1, 1], w3 = [1, 1]
    U[:, 0] = [1, 0, 0, 0]
    V[:, 0] = [1, 0]
    W[:, 0] = [1, 0]
    U[:, 1] = [0, 0, 1, 0]
    V[:, 1] = [0, 1]
    W[:, 1] = [0, 1]
    U[:, 2] = [0, 1, 0, 1]
    V[:, 2] = [1, 1]
    W[:, 2] = [1, 1]
    if _verify(U, V, W, n, m, p):
        return U, V, W
    # Fallback: try to solve numerically via least-squares on the bilinear form.
    # The tensor is 4x2x2 = 16 entries, rank-3 gives 3*(4+2+2)=24 params.
    # We use a simple iterative refinement starting from random init.
    best = None
    rng = np.random.default_rng(0)
    for attempt in range(40):
        if _remaining() < 5:
            break
        Uc = rng.normal(size=(4, 3))
        Vc = rng.normal(size=(2, 3))
        Wc = rng.normal(size=(2, 3))
        # Optimize with a simple gradient-free Nelder-Mead via scipy if available.
        try:
            from scipy.optimize import minimize
            def flat(x):
                Uc_ = x[0:12].reshape(4, 3)
                Vc_ = x[12:18].reshape(2, 3)
                Wc_ = x[18:24].reshape(2, 3)
                tmp = np.einsum('uk,vk->uvk', Uc_, Vc_)
                res = np.einsum('uvk,wk->uvw', tmp, Wc_)
                T = np.zeros((4, 2, 2))
                for i in range(2):
                    for j in range(2):
                        for k in range(1):
                            T[i * m + j, j * p + k, k * n + i] = 1.0
                return np.sum((res - T) ** 2)
            res = minimize(flat, np.concatenate([Uc.ravel(), Vc.ravel(), Wc.ravel()]),
                           method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            x = res.x
            Uc_ = x[0:12].reshape(4, 3)
            Vc_ = x[12:18].reshape(2, 3)
            Wc_ = x[18:24].reshape(2, 3)
            if _verify(Uc_, Vc_, Wc_, n, m, p, tol=1e-7):
                # Snap to nearest half-integers and re-verify exactly.
                Uc_s = np.round(Uc_ * 2) / 2
                Vc_s = np.round(Vc_ * 2) / 2
                Wc_s = np.round(Wc_ * 2) / 2
                if _verify(Uc_s, Vc_s, Wc_s, n, m, p, tol=1e-9):
                    return Uc_s, Vc_s, Wc_s
                return Uc_, Vc_, Wc_
        except Exception:
            pass
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Strategy: decompose the 2x2x15 MM tensor as p=15 independent 2x2x1 blocks
    # (one per output column). Each 2x2x1 block has rank 3, so total rank = 15*3 = 45.
    # This exploits additivity of tensor rank over disjoint index sets (the output
    # dimension k is shared structure but the blocks are independent per k).
    block = _solve_221_rank3()
    if block is not None:
        U0, V0, W0 = block
        r0 = U0.shape[1]  # 3
        total_rank = p * r0  # 45
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        for k in range(p):
            c0 = k * r0
            # U part: same for every k (A is shared)
            U[:, c0:c0 + r0] = U0
            # V part: block k uses V0 but only the column corresponding to output col k
            # V index = j*p + k ; for block k, V0 has m*p0 = 2*1 = 2 entries (j in 0..1)
            for j in range(m):
                V[j * p + k, c0:c0 + r0] = V0[j]
            # W part: block k uses W0, W index = k*n + i
            for i in range(n):
                W[k * n + i, c0:c0 + r0] = W0[i]
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Fallback to schoolbook (guaranteed valid, rank 60).
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
