"""Rank-17 Strassen-based decomposition for 2x3 * 3x3 matmul,
plus a fast linear tensor-completion search for a rank-16 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
U_DIM = N * M   # 6
V_DIM = M * P   # 9
W_DIM = P * N   # 6


def _build_tensor():
    T = np.zeros((U_DIM, V_DIM, W_DIM), dtype=float)
    for i in range(U_DIM):
        ia, ja = divmod(i, M)
        for j in range(V_DIM):
            ja2, kb = divmod(j, P)
            if ja != ja2:
                continue
            for k in range(W_DIM):
                kb2, ic = divmod(k, N)
                if ia == ic and kb == kb2:
                    T[i, j, k] = 1.0
    return T


def _validate(U, V, W, T, atol=1e-10):
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    rank = 17
    U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
    U[0,0]=1;U[4,0]=1;V[0,0]=1;V[4,0]=1;W[0,0]=1;W[3,0]=1
    U[3,1]=1;U[4,1]=1;V[0,1]=1;W[1,1]=1;W[3,1]=-1
    U[0,2]=1;V[1,2]=1;V[4,2]=-1;W[2,2]=1;W[3,2]=1
    U[4,3]=1;V[3,3]=1;V[0,3]=-1;W[0,3]=1;W[1,3]=1
    U[0,4]=1;U[1,4]=1;V[4,4]=1;W[0,4]=-1;W[2,4]=1
    U[3,5]=1;U[0,5]=-1;V[0,5]=1;V[1,5]=1;W[3,5]=1
    U[1,6]=1;U[4,6]=-1;V[3,6]=1;V[4,6]=1;W[0,6]=1
    U[0,7]=1;V[2,7]=1;W[4,7]=1
    U[1,8]=1;V[5,8]=1;W[4,8]=1
    U[3,9]=1;V[2,9]=1;W[5,9]=1
    U[4,10]=1;V[5,10]=1;W[5,10]=1
    U[2,11]=1;V[6,11]=1;W[0,11]=1
    U[2,12]=1;V[7,12]=1;W[2,12]=1
    U[2,13]=1;V[8,13]=1;W[4,13]=1
    U[5,14]=1;V[6,14]=1;W[1,14]=1
    U[5,15]=1;V[7,15]=1;W[3,15]=1
    U[5,16]=1;V[8,16]=1;W[5,16]=1
    return U, V, W


def _err(U, V, W, T):
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return float(np.linalg.norm(recon - T))


def _als_full(U, V, W, T, T0, T1, T2, rank, max_iter, deadline, damp=0.9):
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U = damp * U + (1 - damp) * np.linalg.lstsq(B.T, T0.T, rcond=None)[0]
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V = damp * V + (1 - damp) * np.linalg.lstsq(B.T, T1.T, rcond=None)[0]
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W = damp * W + (1 - damp) * np.linalg.lstsq(B.T, T2.T, rcond=None)[0]
    return U, V, W


def _linear_solve(supU, supV, supW, T, T0, T1, T2):
    """Fix a support, solve the LINEAR system for the free entries.

    Returns (U, V, W) or None if inconsistent / singular.
    """
    rank = 16
    U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
    for (i, r) in supU: U[i, r] = 1.0
    for (j, r) in supV: V[j, r] = 1.0
    for (k, r) in supW: W[k, r] = 1.0

    freeU = {(i, r) for r in range(rank) for i in range(U_DIM)} - set(supU)
    freeV = {(j, r) for r in range(rank) for j in range(V_DIM)} - set(supV)
    freeW = {(k, r) for r in range(rank) for k in range(W_DIM)} - set(supW)

    fvars = [(0, i, r) for (i, r) in sorted(freeU)]
    fvars += [(1, j, r) for (j, r) in sorted(freeV)]
    fvars += [(2, k, r) for (k, r) in sorted(freeW)]
    nf = len(fvars)
    if nf == 0:
        return U, V, W

    # A[ij, f] = coefficient of free var f in tensor entry (i,j,k)
    A = np.zeros((U_DIM * V_DIM * W_DIM, nf), dtype=float)
    b = T.copy()
    for (t, a, r) in fvars:
        f = len(fvars) - nf + fvars.index((t, a, r)) if False else None
    # build A by iterating free vars
    A = np.zeros((U_DIM * V_DIM * W_DIM, nf), dtype=float)
    for f, (t, a, r) in enumerate(fvars):
        if t == 0:
            i = a
            row = V[:, r] * W[:, r]
            if row.any():
                A[i * V_DIM * W_DIM:(i + 1) * V_DIM * W_DIM, f] = row.reshape(-1)
        elif t == 1:
            j = a
            row = U[:, r] * W[:, r]
            if row.any():
                for i in range(U_DIM):
                    A[i * V_DIM * W_DIM + j * W_DIM:(i * V_DIM + j + 1) * W_DIM, f] = row.reshape(-1)
        else:
            k = a
            row = U[:, r] * V[:, r]
            if row.any():
                for i in range(U_DIM):
                    for j in range(V_DIM):
                        A[i * V_DIM * W_DIM + j * W_DIM + k, f] = row[i * W_DIM + j]

    # subtract contributions of fixed (value=1) entries
    for (i, r) in supU:
        row = V[:, r] * W[:, r]
        if row.any():
            b[i * V_DIM * W_DIM:(i + 1) * V_DIM * W_DIM] -= row.reshape(-1)
    for (j, r) in supV:
        row = U[:, r] * W[:, r]
        if row.any():
            for i in range(U_DIM):
                b[i * V_DIM * W_DIM + j * W_DIM:(i * V_DIM + j + 1) * W_DIM] -= row.reshape(-1)
    for (k, r) in supW:
        row = U[:, r] * V[:, r]
        if row.any():
            for i in range(U_DIM):
                for j in range(V_DIM):
                    b[i * V_DIM * W_DIM + j * W_DIM + k] -= row[i * W_DIM + j]

    try:
        x, res, rk, sv = np.linalg.lstsq(A, b, rcond=1e-12)
    except Exception:
        return None
    if rk < nf:
        # singular -> only accept if it already fits
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        if not np.allclose(recon, T, atol=1e-7):
            return None
    # assign
    for f, (t, a, r) in enumerate(fvars):
        if t == 0: U[a, r] = x[f]
        elif t == 1: V[a, r] = x[f]
        else: W[a, r] = x[f]
    return U, V, W


def _refine_support(supU, supV, supW, T, T0, T1, T2, deadline, max_iter=120):
    """Smooth ALS refinement with the support locked, then linear polish."""
    rank = 16
    U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
    for (i, r) in supU: U[i, r] = 1.0
    for (j, r) in supV: V[j, r] = 1.0
    for (k, r) in supW: W[k, r] = 1.0
    supU_s = set(supU); supV_s = set(supV); supW_s = set(supW)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        for (i, r) in supU:
            num = np.sum(T[i] * V[:, r] * W[:, r])
            den = np.sum((V[:, r] ** 2) * (W[:, r] ** 2))
            U[i, r] = num / den if abs(den) > 1e-14 else 0.0
        for (j, r) in supV:
            num = np.sum(T[:, j] * U[:, r] * W[:, r])
            den = np.sum((U[:, r] ** 2) * (W[:, r] ** 2))
            V[j, r] = num / den if abs(den) > 1e-14 else 0.0
        for (k, r) in supW:
            num = np.sum(T[:, :, k] * U[:, r] * V[:, r])
            den = np.sum((U[:, r] ** 2) * (V[:, r] ** 2))
            W[k, r] = num / den if abs(den) > 1e-14 else 0.0
    # linear polish
    res = _linear_solve(supU, supV, supW, T, T0, T1, T2)
    if res is not None:
        U, V, W = res
    return U, V, W


def _try_rank16(T, deadline):
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16
    U17, V17, W17 = _rank17()

    # ---- Strategy A: drop one of the 17 rank-17 columns, linear-completion ----
    for drop in range(17):
        if time.time() > deadline:
            break
        supU = [(i, (c if c < drop else c - 1)) for c in range(17) if c != drop
                for i in range(U_DIM) if abs(U17[i, c]) > 1e-10]
        supV = [(j, (c if c < drop else c - 1)) for c in range(17) if c != drop
                for j in range(V_DIM) if abs(V17[j, c]) > 1e-10]
        supW = [(k, (c if c < drop else c - 1)) for c in range(17) if c != drop
                for k in range(W_DIM) if abs(W17[k, c]) > 1e-10]
        res = _linear_solve(supU, supV, supW, T, T0, T1, T2)
        if res is not None and _validate(res[0], res[1], res[2], T, atol=1e-8):
            return res
        # refine this support
        U, V, W = _refine_support(supU, supV, supW, T, T0, T1, T2, deadline, max_iter=60)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # ---- Strategy B: smooth multi-start full-rank ALS, then support-locked polish ----
    best = None
    best_err = float('inf')
    for trial in range(30):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(777 + trial * 131)
        if trial == 0:
            U = U17[:, :16].copy(); V = V17[:, :16].copy(); W = W17[:, :16].copy()
        elif trial == 1:
            U = U17[:, 1:].copy(); V = V17[:, 1:].copy(); W = W17[:, 1:].copy()
        else:
            U = U17[:, :16] * rng.randn(U_DIM, 16) * 0.3
            V = V17[:, :16] * rng.randn(V_DIM, 16) * 0.3
            W = W17[:, :16] * rng.randn(W_DIM, 16) * 0.3
        U, V, W = _als_full(U, V, W, T, T0, T1, T2, rank, max_iter=60, deadline=deadline)
        e = _err(U, V, W, T)
        if e < best_err:
            best_err = e; best = (U.copy(), V.copy(), W.copy())
        if e < 1e-8:
            return U, V, W
        # support-locked polish from best so far
        if e < 2.0:
            thr = max(1e-7, 0.5 * np.percentile(np.abs(U), 50))
            supU = [(i, r) for r in range(rank) for i in range(U_DIM) if abs(U[i, r]) > thr]
            supV = [(j, r) for r in range(rank) for j in range(V_DIM) if abs(V[j, r]) > thr]
            supW = [(k, r) for r in range(rank) for k in range(W_DIM) if abs(W[k, r]) > thr]
            U2, V2, W2 = _refine_support(supU, supV, supW, T, T0, T1, T2, deadline, max_iter=80)
            e2 = _err(U2, V2, W2, T)
            if e2 < best_err:
                best_err = e2; best = (U2.copy(), V2.copy(), W2.copy())
            if e2 < 1e-8:
                return U2, V2, W2

    if best is not None and _validate(best[0], best[1], best[2], T, atol=1e-8):
        return best
    return None


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill

    U17, V17, W17 = _rank17()
    T = _build_tensor()
    base = (U17, V17, W17)
    if not _validate(U17, V17, W17, T):
        rank = N * M * P
        U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
        col = 0
        for ia in range(N):
            for ja in range(M):
                for kb in range(P):
                    U[ia * M + ja, col] = 1.0
                    V[ja * P + kb, col] = 1.0
                    W[kb * N + ia, col] = 1.0
                    col += 1
        base = (U, V, W)

    try:
        r16 = _try_rank16(T, deadline)
        if r16 is not None and _validate(r16[0], r16[1], r16[2], T, atol=1e-8):
            base = r16
    except Exception:
        pass

    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
