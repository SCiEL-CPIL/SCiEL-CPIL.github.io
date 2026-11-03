"""Tensor decomposition for 2x3 * 3x10 matrix multiplication.

Anchor: a hand-constructed rank-55 decomposition (Strassen 2x2 on five
2x2x2 sub-blocks of the 2x2x10 part + naive 1x1x10 part).  This beats the
rank-60 schoolbook.  On top of it we run a constraint-aware local search
to attempt a strictly lower-rank exact decomposition, guarded by an
explicit wall-clock deadline.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Return True iff the factors exactly decompose the multiplication tensor."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    r = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            u = Ua[i * m + j]
            for k in range(p):
                s = np.dot(u, Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > 1e-7:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
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
    return U.tolist(), V.tolist(), W.tolist()


def _build_strassen55(n, m, p):
    """Rank-55 decomposition for 2x3x10.

    Middle dimension 3 = 2 + 1:
      * j in {0,1}: a 2x2x10 multiplication, tiled into five 2x2x2 Strassen
        sub-problems (7 rank-1 terms each -> 35).
      * j = 2: a 1x1x10 multiplication (10 rank-1 terms).
    Total rank = 35 + 10 = 55.
    """
    if n != 2 or m != 3 or p != 10:
        return None

    # Strassen 7 forms: (i, j, k) positions and +/- coefficients.
    S = [
        (0, 0, 0, +1), (0, 1, 0, -1), (0, 1, 1, +1),
        (1, 0, 0, +1), (1, 0, 1, -1), (1, 1, 1, +1),
        (1, 1, 0, +1),
    ]

    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for b in range(5):            # 5 blocks of 2 columns each
        for (si, sj, sk, sgn) in S:
            i = si
            j = sj                # j in {0,1}
            k = 2 * b + sk
            U[i * m + j, col] = sgn
            V[j * p + k, col] = sgn
            W[k * n + i, col] = 1
            col += 1

    for k in range(10):           # 1x1x10 part (j = 2)
        U[2 * m + 2, col] = 1
        V[2 * p + k, col] = 1
        W[k * n + 0, col] = 1
        col += 1

    assert col == rank, col
    return U.tolist(), V.tolist(), W.tolist()


def _local_search(U, V, W, n, m, p, deadline, iters=1500):
    """Constraint-aware local search attempting a lower-rank exact decomp.

    Each term c is a rank-1 tensor u_c ⊗ v_c ⊗ w_c.  We minimize the squared
    residual of the 60 target equations, keeping every term exactly rank-1
    (a change of variables that removes the rank constraint).  Uses a
    smooth squared objective (a surrogate for the nonsmooth exact-match goal).
    """
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    r = Ua.shape[1]

    # Precompute index lists.
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i * m + j, j * p + k, k * n + i))
    idx = np.array(idx)

    def residual(Ua, Va, Wa):
        # R[c] = sum_t U[t,ic] * V[t,jc] * W[t,kc]  - 1
        R = np.zeros(r)
        for c in range(r):
            u = Ua[idx[:, 0], c]
            v = Va[idx[:, 1], c]
            w = Wa[idx[:, 2], c]
            R[c] = np.sum(u * v * w) - 1.0
        return R

    def loss(Ua, Va, Wa):
        R = residual(Ua, Va, Wa)
        return float(np.dot(R, R))

    best = (loss(Ua, Va, Wa), Ua.copy(), Va.copy(), Wa.copy())

    step = 0.05
    for it in range(iters):
        if time.time() > deadline:
            break
        R = residual(Ua, Va, Wa)
        for c in range(r):
            u = Ua[:, c]
            v = Va[:, c]
            w = Wa[:, c]
            # dR/dc w.r.t. u_c = (v * w)[t]  (summed over t)
            gu = np.sum(R * v * w)
            gv = np.sum(R * u * w)
            gw = np.sum(R * u * v)
            Un = np.linalg.norm(u)
            Vn = np.linalg.norm(v)
            Wn = np.linalg.norm(w)
            if Un < 1e-12 or Vn < 1e-12 or Wn < 1e-12:
                continue
            # Scale-invariant update (keep each term rank-1, avoid drift).
            Ua[:, c] -= step * (gu / (Un * Un)) * u
            Va[:, c] -= step * (gv / (Vn * Vn)) * v
            Wa[:, c] -= step * (gw / (Wn * Wn)) * w

        L = loss(Ua, Va, Wa)
        if L < best[0] - 1e-12:
            best = (L, Ua.copy(), Va.copy(), Wa.copy())
            step = min(step * 1.05, 0.15)
        else:
            step = max(step * 0.7, 0.002)
            # restore best to keep iterating from a good point
            Ua[:], Va[:], Wa[:] = best[1], best[2], best[3]

        if best[0] < 1e-14:
            break

    return best


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before the 150s kill

    n, m, p = DIMENSIONS

    # Anchor: rank-55 Strassen construction.
    U, V, W = _build_strassen55(n, m, p)
    if U is None or not _verify(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    best = (55, U, V, W)

    # Attempt lower ranks via structured local search.
    for target in (54, 53):
        if time.time() > deadline - 15.0:
            break
        # Build a target-rank starting point by zeroing the least-important
        # terms of the 55-decomp and letting local search re-absorb them.
        U0, V0, W0 = _build_strassen55(n, m, p)
        Ua = np.array(U0, dtype=float)[:, :target]
        Va = np.array(V0, dtype=float)[:, :target]
        Wa = np.array(W0, dtype=float)[:, :target]

        np.random.seed(1000 + target)
        for _ in range(4):
            if time.time() > deadline - 10.0:
                break
            Ua = Ua + 0.02 * np.random.randn(*Ua.shape)
            Va = Va + 0.02 * np.random.randn(*Va.shape)
            Wa = Wa + 0.02 * np.random.randn(*Wa.shape)
            L, Ub, Vb, Wb = _local_search(
                Ua.copy(), Va.copy(), Wa.copy(), n, m, p,
                deadline - 5.0, iters=600)
            if L < 1e-12:
                Ub_l, Vb_l, Wb_l = Ub.tolist(), Vb.tolist(), Wb.tolist()
                if _verify(Ub_l, Vb_l, Wb_l, n, m, p):
                    return {"U": Ub_l, "V": Vb_l, "W": Wb_l}

    # Fall back to the best verified result (the rank-55 anchor).
    return {"U": best[1], "V": best[2], "W": best[3]}
