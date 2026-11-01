"""Rewrite for 2x3x8 multiplication tensor: block-structured search for a rank-minimal
exact decomposition, with a deadline guard and a hard fallback to the schoolbook rank."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 48

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = time.time() + 135.0


def _target_tensor():
    """Build the multiplication tensor T[i, j, k] = 1 (the 2x3x8 multiplication tensor)."""
    T = np.ones((N, M, P), dtype=float)
    return T


def _verify(U, V, W, T):
    """Check sum_c U[i,j,c] V[j,k,c] W[k,i,c] == T[i,j,k]."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # U: (n*m, r) -> reshape (n, m, r); V: (m*p, r) -> (m, p, r); W: (p*n, r) -> (p, n, r)
    r = U.shape[1]
    Ur = U.reshape(N, M, r)
    Vr = V.reshape(M, P, r)
    Wr = W.reshape(P, N, r)
    # result[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # contract: A[i,j,k] = sum_c Ur[i,j,c]*Vr[j,k,c] -> (n,m,p) then * Wr
    A = np.einsum('ijc,jkc->ijk', Ur, Vr)
    B = np.einsum('kic->ikc', Wr)  # (p,n,r) -> (i,k,c)
    res = np.einsum('ijk,ikc->ijk', A, B)
    return np.max(np.abs(res - T)) < 1e-6


def _als_search(T, r, n_iter=40, n_restart=1, seed=0):
    """Alternating least squares to fit T with rank r. Returns (U,V,W) or None."""
    rng = np.random.default_rng(seed)
    best_err = np.inf
    best = None
    for restart in range(n_restart):
        # random init
        Ur = rng.standard_normal((N, M, r))
        Vr = rng.standard_normal((M, P, r))
        Wr = rng.standard_normal((P, N, r))
        prev = np.inf
        for it in range(n_iter):
            # update Ur: for each (i,j): solve sum_k Vr[j,k,:] Wr[k,i,:] @ Ur[i,j,:] = T[i,j,k]
            # build matrix G[i,j] (r x r): G = sum_k Vr[j,k,:,None]*Wr[k,i,None,:]
            G = np.einsum('jkc,kic->ijc', Vr, Wr)  # (n,m,r,r)
            b = T  # (n,m,p)
            # Ur[i,j,c] = (G[i,j]^{-1}) @ b[i,j]
            try:
                Ur = np.einsum('ijc,ikc->ijk', G, b)
                Ur = np.linalg.solve(G, b[..., None]).squeeze(-1)
            except np.linalg.LinAlgError:
                Ur = np.linalg.lstsq(G, b[..., None], rcond=None)[0].squeeze(-1)
            # update Vr: for each (j,k): sum_i Ur[i,j,i] Wr[k,i,:] ... 
            # Vr[j,k,c]: sum_i Ur[i,j,c] Wr[k,i,:] -> but need sum_i Ur[i,j,:] Wr[k,i,:] as r x r
            G2 = np.einsum('ijc,kic->jkci', Ur, Wr)  # (m,p,r,r)
            b2 = T.transpose(1, 2, 0)  # (m,p,n)
            try:
                Vr = np.linalg.solve(G2, b2[..., None]).squeeze(-1)
            except np.linalg.LinAlgError:
                Vr = np.linalg.lstsq(G2, b2[..., None], rcond=None)[0].squeeze(-1)
            # update Wr: for each (k,i): sum_j Ur[i,j,:] Vr[j,k,:]
            G3 = np.einsum('ijc,jkc->kici', Ur, Vr)  # (p,n,r,r)
            b3 = T.transpose(2, 0, 1)  # (p,n,m)
            try:
                Wr = np.linalg.solve(G3, b3[..., None]).squeeze(-1)
            except np.linalg.LinAlgError:
                Wr = np.linalg.lstsq(G3, b3[..., None], rcond=None)[0].squeeze(-1)
            # error
            A = np.einsum('ijc,jkc->ijk', Ur, Vr)
            B = np.einsum('kic->ikc', Wr)
            res = np.einsum('ijk,ikc->ijk', A, B)
            err = np.max(np.abs(res - T))
            if err < best_err:
                best_err = err
                best = (Ur.copy(), Vr.copy(), Wr.copy())
            if err < 1e-6:
                break
            if time.time() > _DEADLINE:
                break
        if time.time() > _DEADLINE:
            break
    if best is not None and best_err < 1e-6:
        Ur, Vr, Wr = best
        U = Ur.reshape(N * M, r)
        V = Vr.reshape(M * P, r)
        W = Wr.reshape(P * N, r)
        return U, V, W
    return None


def _schoolbook():
    rank = RANK_SCHOOLBOOK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    T = _target_tensor()
    # Try a sequence of candidate ranks from low to high.
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6,24,16)=24.
    # We probe a few ranks; if none work, fall back to schoolbook.
    for r in [24, 25, 26, 27, 28, 30, 32, 36, 40]:
        if time.time() > _DEADLINE:
            break
        res = _als_search(T, r, n_iter=30, n_restart=2, seed=r)
        if res is not None:
            U, V, W = res
            if _verify(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    # Fallback: guaranteed-valid schoolbook decomposition.
    return _schoolbook()
