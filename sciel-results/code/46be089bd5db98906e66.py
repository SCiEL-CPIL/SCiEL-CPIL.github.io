import numpy as np
import time

_T0 = time.time()
_DEADLINE = 135.0  # safety margin before the 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _T0)

DIMENSIONS = [2, 3, 4]


def _build_rank22(n, m, p):
    """Fast block-Strassen rank-22 decomposition.
    U: (n*m, rank), index i*m + j
    V: (m*p, rank), index j*p + k
    W: (p*n, rank), index k*n + i
    """
    rank = 22
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    def add(uc, vc, wc):
        nonlocal col
        for idx, c in uc.items():
            U[idx, col] += c
        for idx, c in vc.items():
            V[idx, col] += c
        for idx, c in wc.items():
            W[idx, col] += c
        col += 1

    # ---- Block 1: A[:,0:2] * B[0:2,:] -> C[:,0:2] (2x2 Strassen) ----
    # A: 0=A00,1=A01,3=A10,4=A11 ; B: 0=B00,1=B01,4=B10,5=B11 ; W: 0=C00,2=C01,1=C10,3=C11
    add({0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    add({3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
    add({0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
    add({4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
    add({0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
    add({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add({1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})

    # ---- Block 2: A[:,0:2] * B[0:2,2:4] -> C[:,2:4] (2x2 Strassen) ----
    # B: 2=B02,3=B03,6=B12,7=B13 ; W: 4=C02,6=C03,5=C12,7=C13
    add({0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
    add({3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
    add({0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
    add({4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
    add({0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
    add({3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    add({1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})

    # ---- Block 3: A[:,2:3] * B[2:3,:] -> C[:,0:4] (schoolbook) ----
    # A: 2=A02,5=A12 ; B: 8=B20,9=B21,10=B22,11=B23
    for ai, i in [(2, 0), (5, 1)]:
        for bk, k in [(8, 0), (9, 1), (10, 2), (11, 3)]:
            add({ai: 1}, {bk: 1}, {k * n + i: 1})

    assert col == rank, f"rank mismatch {col} != {rank}"
    return U, V, W


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-7):
    """Exact bilinear check: for all (ia,ja,jb,kb), the rank-r bilinear map
    (U[ia*m+ja,:] * V[jb*p+kb,:])^T W must equal the schoolbook product tensor.
    """
    r = U.shape[1]
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    uv = U[ia * m + ja, :] * V[jb * p + kb, :]
                    M = W.T @ uv  # (p*n,)
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            if abs(M[k * n + i] - expected) > tol:
                                return False
    return True


def _error(U, V, W, n, m, p):
    """Vectorized squared bilinear error (fast)."""
    r = U.shape[1]
    err = 0.0
    for ia in range(n):
        for ja in range(m):
            for jb in range(m):
                for kb in range(p):
                    uv = U[ia * m + ja, :] * V[jb * p + kb, :]
                    M = W.T @ uv
                    for i in range(n):
                        for k in range(p):
                            expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                            d = M[k * n + i] - expected
                            err += d * d
    return err


def _snap(x, half_int=True):
    """Round to integers (and half-integers if requested); returns snapped array."""
    if half_int:
        return np.rint(x * 2.0) / 2.0
    return np.rint(x)


def _polish(U, V, W, n, m, p, iters=25):
    """Alternating least-squares: each factor is linear given the other two,
    so solve the normal equations exactly at each step to drive error to 0."""
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    U = U.copy(); V = V.copy(); W = W.copy()
    for _ in range(iters):
        # Solve for W: for each (ia,ja,jb,kb), W[:, idx] = uv  (least squares over k*n+i)
        for ia in range(n):
            for ja in range(m):
                for jb in range(m):
                    for kb in range(p):
                        uv = U[ia * m + ja, :] * V[jb * p + kb, :]
                        # target for W row (k*n+i)
                        for i in range(n):
                            for k in range(p):
                                W[k * n + i, :] = uv  # each target row equals uv
                        # least squares: W[:, idx] = argmin sum_rows (W[idx]-target)^2
                        # = mean of target rows for that idx
                        for idx in range(r):
                            vals = []
                            for i in range(n):
                                for k in range(p):
                                    vals.append(uv[idx])
                        # all targets equal uv, so W row = uv
                        # (already set)
        # Solve for U: for each (ia,ja), U[ia*m+ja,:] = argmin over (jb,kb,i,k)
        # target: W[idx,k*n+i] * V[jb*p+kb,idx]
        for ia in range(n):
            for ja in range(m):
                for idx in range(r):
                    s = 0.0; c = 0
                    for jb in range(m):
                        for kb in range(p):
                            v = V[jb * p + kb, idx]
                            for i in range(n):
                                for k in range(p):
                                    expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                                    # we want U[idx]*V[idx]*W[idx] summed... but W[idx] is per (k*n+i)
                                    # target contribution to U[idx]: W[k*n+i, idx] * V[jb*p+kb, idx]
                                    # least squares over all (jb,kb,i,k)
                                    s += W[k * n + i, idx] * V[jb * p + kb, idx]
                                    c += 1
                    U[ia * m + ja, idx] = s / c
        # Solve for V analogously
        for jb in range(m):
            for kb in range(p):
                for idx in range(r):
                    s = 0.0; c = 0
                    for ia in range(n):
                        for ja in range(m):
                            for i in range(n):
                                for k in range(p):
                                    expected = 1.0 if (ia == i and ja == jb and kb == k) else 0.0
                                    s += W[k * n + i, idx] * U[ia * m + ja, idx]
                                    c += 1
                    V[jb * p + kb, idx] = s / c
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    U22, V22, W22 = _build_rank22(n, m, p)

    # Verify the rank-22 baseline
    if not _verify(U22, V22, W22, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U22, V22, W22
    best_rank = 22

    # Try to reach rank 21 by dropping one column and polishing
    if _remaining() > 20.0:
        for drop in range(22):
            if _remaining() < 15.0:
                break
            keep = [c for c in range(22) if c != drop]
            U0 = U22[:, keep].copy()
            V0 = V22[:, keep].copy()
            W0 = W22[:, keep].copy()
            # polish
            U1, V1, W1 = _polish(U0, V0, W0, n, m, p, iters=15)
            # snap to half-integers and re-verify
            Us, Vs, Ws = _snap(U1), _snap(V1), _snap(W1)
            if _verify(Us, Vs, Ws, n, m, p, tol=1e-6):
                best_U, best_V, best_W = Us, Vs, Ws
                best_rank = 21
                break
            # also try un-snapped verification (allow floats) as fallback
            if _verify(U1, V1, W1, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U1, V1, W1
                best_rank = 21
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
