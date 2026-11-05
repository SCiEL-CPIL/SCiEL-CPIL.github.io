"""Rank-16 search for 2x3 * 3x3 matmul via structured perturbation of the rank-17 seed,
plus a fallback to the verified rank-17 decomposition. The rank-16 attempt uses a
deadline-guarded local optimisation (ALS + random restarts) starting from the rank-17
structure with one column removed and refined, as well as from structured random
initialisations that respect the tensor's support pattern."""
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
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    rank = 17
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[5, 14] = 1; V[6, 14] = 1; W[1, 14] = 1
    U[5, 15] = 1; V[7, 15] = 1; W[3, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    return U, V, W


def _als_step(U, V, W, T0, T1, T2, rank, max_iter=200, deadline=None):
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
    return U, V, W, True


def _structured_init(rank, rng, T):
    """Initialise factor matrices with structure: each column gets a small number of
    non-zero entries chosen to cover the tensor's support."""
    nz = np.argwhere(T != 0)
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    for c in range(rank):
        # Pick 3 random support entries to seed this column
        for _ in range(3):
            idx = rng.randint(len(nz))
            i, j, k = nz[idx]
            U[i, c] = rng.randn()
            V[j, c] = rng.randn()
            W[k, c] = rng.randn()
    return U, V, W


def _polish(U, V, W, T, T0, T1, T2, rank, deadline, max_iter=500):
    """Final high-precision polish: run ALS with tighter convergence and check."""
    U, V, W, _ = _als_step(U, V, W, T0, T1, T2, rank, max_iter=max_iter, deadline=deadline)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    err = np.linalg.norm(recon - T)
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False


def _try_rank16(deadline):
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16

    # Strategy 1: Start from rank-17, drop one column, refine + polish
    U17, V17, W17 = _rank17()
    best_err = float('inf')
    best_result = None
    for drop_col in range(17):
        if time.time() > deadline:
            break
        keep = [c for c in range(17) if c != drop_col]
        U = U17[:, keep].copy()
        V = V17[:, keep].copy()
        W = W17[:, keep].copy()
        U, V, W, _ = _als_step(U, V, W, T0, T1, T2, rank, max_iter=100, deadline=deadline)
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        if err < 1e-8:
            U, V, W, ok = _polish(U, V, W, T, T0, T1, T2, rank, deadline)
            if ok:
                return U, V, W
            if _validate(U, V, W, T, atol=1e-8):
                return U, V, W

    # Strategy 2: Structured random initialisations + ALS + polish
    n_restarts = 40
    for trial in range(n_restarts):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(12345 + trial * 7919)
        U, V, W = _structured_init(rank, rng, T)
        U, V, W, _ = _als_step(U, V, W, T0, T1, T2, rank, max_iter=80, deadline=deadline)
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        if err < 1e-8:
            U, V, W, ok = _polish(U, V, W, T, T0, T1, T2, rank, deadline)
            if ok:
                return U, V, W
            if _validate(U, V, W, T, atol=1e-8):
                return U, V, W

    # Strategy 3: Start from rank-17, perturb all columns slightly, refine + polish
    for perturb_scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
        if time.time() > deadline:
            break
        rng = np.random.RandomState(99999 + int(perturb_scale * 100))
        U = U17[:, :rank].copy() + rng.randn(U_DIM, rank) * perturb_scale
        V = V17[:, :rank].copy() + rng.randn(V_DIM, rank) * perturb_scale
        W = W17[:, :rank].copy() + rng.randn(W_DIM, rank) * perturb_scale
        U, V, W, _ = _als_step(U, V, W, T0, T1, T2, rank, max_iter=100, deadline=deadline)
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        if err < 1e-8:
            U, V, W, ok = _polish(U, V, W, T, T0, T1, T2, rank, deadline)
            if ok:
                return U, V, W
            if _validate(U, V, W, T, atol=1e-8):
                return U, V, W

    # Final check on best result found
    if best_result is not None:
        U, V, W = best_result
        U, V, W, ok = _polish(U, V, W, T, T0, T1, T2, rank, deadline)
        if ok or _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    return None


def construct():
    start = time.time()
    deadline = start + 135

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
        r16 = _try_rank16(deadline)
        if r16 is not None:
            U16, V16, W16 = r16
            if _validate(U16, V16, W16, T, atol=1e-8):
                base = r16
    except Exception:
        pass

    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
