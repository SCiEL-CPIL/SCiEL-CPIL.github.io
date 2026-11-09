import numpy as np
import time

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

_DEADLINE_START = None
_TIME_BUDGET = 135.0  # keep 15s safety margin before the 150s kill


def _elapsed():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START


def _within_budget(margin=5.0):
    return _elapsed() < _TIME_BUDGET - margin


# Strassen's 7 factors for 2x2x2.
# Each: (u_entries, v_entries, w_entries)
#   u_entries: [(i, j_local, coeff), ...]
#   v_entries: [(j_local, k_local, coeff), ...]
#   w_entries: [(k_local, i, coeff), ...]
_STRASSEN_7 = [
    ([(0, 0, 1), (1, 1, 1)],
     [(0, 0, 1), (1, 1, 1)],
     [(0, 0, 1), (1, 1, 1)]),
    ([(1, 0, 1), (1, 1, 1)],
     [(0, 0, 1)],
     [(0, 1, 1), (1, 1, -1)]),
    ([(0, 0, 1)],
     [(0, 1, 1), (1, 1, -1)],
     [(1, 0, 1), (1, 1, 1)]),
    ([(1, 1, 1)],
     [(1, 0, 1), (0, 0, -1)],
     [(0, 0, 1), (0, 1, 1)]),
    ([(0, 0, 1), (0, 1, 1)],
     [(1, 1, 1)],
     [(0, 0, -1), (1, 0, 1)]),
    ([(1, 0, 1), (0, 0, -1)],
     [(0, 0, 1), (0, 1, 1)],
     [(1, 1, 1)]),
    ([(0, 1, 1), (1, 1, -1)],
     [(1, 0, 1), (1, 1, 1)],
     [(0, 0, 1)]),
]


def _entries_to_matrices(Ue, Ve, We, n, m, p, rank):
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for (f, r, v) in Ue:
        U[r, f] += v
    for (f, r, v) in Ve:
        V[r, f] += v
    for (f, r, v) in We:
        W[r, f] += v
    return U, V, W


def _build_strassen_2x5x5():
    """Deterministic rank-46 construction (verified)."""
    n, m, p = N, M, P
    j_groups = [[0, 1], [2, 3], [4]]
    k_groups = [[0, 1], [2, 3], [4]]

    Ue, Ve, We = [], [], []
    fid = 0
    for a in range(3):
        for b in range(3):
            jg = j_groups[a]
            kg = k_groups[b]
            sj, sk = len(jg), len(kg)
            if sj == 2 and sk == 2:
                for (u_ent, v_ent, w_ent) in _STRASSEN_7:
                    for (i, jl, c) in u_ent:
                        Ue.append((fid, i * m + jg[jl], c))
                    for (jl, kl, c) in v_ent:
                        Ve.append((fid, jg[jl] * p + kg[kl], c))
                    for (kl, i, c) in w_ent:
                        We.append((fid, kg[kl] * n + i, c))
                    fid += 1
            else:
                for i in range(n):
                    for jl in range(sj):
                        for kl in range(sk):
                            Ue.append((fid, i * m + jg[jl], 1))
                            Ve.append((fid, jg[jl] * p + kg[kl], 1))
                            We.append((fid, kg[kl] * n + i, 1))
                            fid += 1
    return _entries_to_matrices(Ue, Ve, We, n, m, p, fid)


def _build_schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, num_tests=5, tol=1e-9):
    """Verify T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] on random slices."""
    for test in range(num_tests):
        rng = np.random.RandomState(1000 + test)
        A = rng.randn(n * m)
        B = rng.randn(m * p)
        A_mat = A.reshape(n, m)
        B_mat = B.reshape(m, p)
        C_true = (A_mat @ B_mat).flatten()
        C_true_wn = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_true_wn[k * n + i] = C_true[i * p + k]
        C_comp = np.zeros(p * n)
        for c in range(U.shape[1]):
            vA = float(U[:, c] @ A)
            vB = float(V[:, c] @ B)
            if vA != 0.0 and vB != 0.0:
                C_comp += W[:, c] * vA * vB
        if not np.allclose(C_comp, C_true_wn, atol=tol):
            return False
    return True


def _refine_scalar_into_strassen(U, V, W, n, m, p):
    """
    Attempt to fold the single redundant 1x1 scalar term into the Strassen
    factors. The 1x1 block (j=4,k=4) is the scalar product sum_i A[i,4]*B[4,k].
    If that linear functional is in the span of existing U/V columns, we can
    complete W and drop the extra term -> rank 45.
    """
    # Identify the redundant scalar column: it is the last one added (fid-1)
    # Build the target residual tensor R[a,b,c] for the scalar term:
    #   R[a,b,c] = 1 iff a = i*m+4, b = 4*p+k, c = k*n+i
    n, m, p = n, m, p
    # We treat the last column as the scalar term.
    r = U.shape[1]
    U0, V0, W0 = U[:, :-1], V[:, :-1], W[:, :-1]
    U1, V1, W1 = U[:, -1], V[:, -1], W[:, -1]

    # Target: T1[a,b,c] = U1[a]*V1[b]*W1[c]
    # We want T1 = sum_{s<r0} U0[a,s]*V0[b,s]*W0[c,s] + U0[a,:]*V0[b,:]*Wc
    # i.e. find Wc (length r0) minimizing ||T1 - U0 V0^T Wc|| over (a,b,c).
    # Vectorize over (a,b): for each (a,b), residual vector over c.
    r0 = U0.shape[1]
    # M[a,b] = U0[a,:] * V0[b,:]  (elementwise) -> shape (nm, mp, r0)
    M = U0[:, :, None] * V0[None, :, :]  # (nm, mp, r0)
    T1 = U1[:, None, None] * V1[None, :, None] * W1[None, None, :]  # (nm, mp, pn)
    # Solve for Wc: for each (a,b), M[a,b] @ Wc = T1[a,b]
    # Stack: A_mat (nm*mp, r0), rhs (nm*mp, pn)
    A_mat = M.reshape(-1, r0)
    rhs = T1.reshape(-1, p * n)
    try:
        Wc, *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    Wc = Wc.reshape(-1, p * n)  # (nm*mp, pn)
    # Check residual
    recon = np.einsum('abx,ab->abc', M, Wc)  # not quite; need per (a,b)
    # Recompute properly: recon[a,b,c] = sum_s M[a,b,s]*Wc[(a*mp+b),c]
    recon = np.zeros((n * m, m * p, p * n), dtype=float)
    for a in range(n * m):
        for b in range(m * p):
            recon[a, b, :] = M[a, b, :] @ Wc[a * m * p + b, :]
    resid = np.max(np.abs(recon - T1))
    if resid < 1e-9:
        # Success: new W = [W0, Wc flattened to columns]
        # Wc is (nm*mp, pn); we need W_new[c, s] for s in [0, r0)
        # W_new[:, s] = Wc[:, s]
        W_new = Wc.T  # (pn, r0)
        return U0, V0, W_new
    return None


def _try_3x3_strassen_4x4():
    """
    Embed 4x4 multiplication as a 3x3 Strassen (rank 7) over 2x2 blocks,
    then place it into the 2x5x5 problem on j,k in {0,1,2,3}.
    This gives rank 7 (for the 4x4 part) + 4 (2x4 schoolbook j=4) + 4 (4x2 k=4) + 1 (1x1)
    = 7 + 4 + 4 + 1 = 16 ... but the 2x2 block Strassen requires the full
    2x2 A-block which is 2x2 (good, n=2) and 2x2 B-block.
    However the 4x4 result is 4x4 while our C is 2x5, so the 4x4 output
    rows beyond n=2 do not exist -> this does NOT directly apply. Return None.
    """
    return None


def construct():
    n, m, p = N, M, P

    # 1) Deterministic rank-46 construction (guaranteed valid).
    U46, V46, W46 = _build_strassen_2x5x5()
    if _verify(U46, V46, W46, n, m, p, num_tests=5):
        best = (46, U46, V46, W46)
    else:
        best = None

    # 2) Try to refine: fold the redundant scalar term into Strassen factors.
    if best is not None and _within_budget(margin=10.0):
        refined = _refine_scalar_into_strassen(U46, V46, W46, n, m, p)
        if refined is not None:
            Ur, Vr, Wr = refined
            if _verify(Ur, Vr, Wr, n, m, p, num_tests=5):
                best = (Ur.shape[1], Ur, Vr, Wr)

    # 3) 3x3 Strassen embedding of 4x4 (does not apply directly, but keep hook).
    if best is not None and _within_budget(margin=10.0):
        alt = _try_3x3_strassen_4x4()
        if alt is not None:
            Ua, Va, Wa = alt
            if _verify(Ua, Va, Wa, n, m, p, num_tests=5):
                if Ua.shape[1] < best[0]:
                    best = (Ua.shape[1], Ua, Va, Wa)

    # 4) Fallback to schoolbook rank-50 if nothing valid.
    if best is None:
        Us, Vs, Ws = _build_schoolbook()
        best = (50, Us, Vs, Ws)

    r, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
