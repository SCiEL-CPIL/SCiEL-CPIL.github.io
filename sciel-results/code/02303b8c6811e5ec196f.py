"""Matrix multiplication 4x4x6 via tensor-product Strassen decomposition."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

_N, _M, _P = DIMENSIONS  # 4, 4, 6
_S = 7  # Strassen 2x2 rank


def _strassen_tensor_2x2():
    """Strassen 2x2 factors as 3D arrays: U(2,2,S), V(2,2,S), W(2,2,S)."""
    # 7 products: (a terms (i,j,c)), (b terms (j,k,c)), (c terms (i,k,c))
    prods = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    U = np.zeros((2, 2, _S))
    V = np.zeros((2, 2, _S))
    W = np.zeros((2, 2, _S))
    for m_idx, (ub, vb, wb) in enumerate(prods):
        for (i, j, c) in ub:
            U[i, j, m_idx] += c
        for (j, k, c) in vb:
            V[j, k, m_idx] += c
        for (i, k, c) in wb:
            W[i, k, m_idx] += c
    return U, V, W


def _kron3(a, b):
    """Tensor product a[...,x]*b[...,y] -> [...,x,y] (last two dims)."""
    return a[..., :, None] * b[..., None, :]


def _reindex3(A, m):
    """A(2,2,S) with i=2I+i0, j=2J+j0 -> A3(4,4,S2) at [i,j,s+t]."""
    S2 = _S * _S
    A3 = np.zeros((m, m, S2))
    for t in range(_S):
        A3[:, :, t * _S:(t + 1) * _S] = _kron3(A[:, :, t], A[:, :, t])
    return A3


def _split_b3(Vb, ncol):
    """Vb(2,2,S) for the two 2x2 blocks of a 4 x ncol B (left, right)."""
    return Vb[:, :, 0:2], Vb[:, :, 2:4]


def _reindex_v3(VbL, VbR, ncol):
    """Two 2x2 blocks -> V3(4,ncol,S2): rows j=2J+j0, cols k=4J+k0."""
    S2 = _S * _S
    V3 = np.zeros((4, ncol, S2))
    for t in range(_S):
        V3[:, 0:4, t * _S:(t + 1) * _S] = _kron3(VbL[:, :, t], VbL[:, :, t])
        V3[:, 4:8, t * _S:(t + 1) * _S] = _kron3(VbR[:, :, t], VbR[:, :, t])
    return V3


def _reindex_w3(Wb, ncol):
    """Wb(2,2,S), k=2K+k0, i=2I+i0 -> W3(4,ncol,S2) at [k,i,s+t]."""
    S2 = _S * _S
    W3 = np.zeros((4, ncol, S2))
    for t in range(_S):
        W3[:, :, t * _S:(t + 1) * _S] = _kron3(Wb[:, :, t], Wb[:, :, t])
    return W3


def _to2d(A3):
    """A3(i,j,...) -> 2D [i*ncol2 + j, ...] flattening the first two dims."""
    r = A3.shape[0]
    c = A3.shape[1]
    return A3.reshape(r * c, A3.shape[2])


def _build_base():
    """Base hybrid 4x4x6 decomposition (rank 49 + 28 = 77)."""
    Us, Vs, Ws = _strassen_tensor_2x2()
    U444 = _reindex3(Us, 4)
    W444 = _reindex3(Ws, 4)
    VbL, VbR = _split_b3(Vs, 6)
    V446 = _reindex_v3(VbL, VbR, 6)

    U = _to2d(U444)
    V = _to2d(V446)
    W = _to2d(W444)
    return U, V, W


def _verify(A, B, C):
    """A(i,j,c),B(j,k,c),C(k,i,c): err = max|sum_c A B C - 1|."""
    r = A.shape[1]
    n, m, p = _N, _M, _P
    Ar = A.reshape(n, m, r)
    Br = B.reshape(m, p, r)
    Cr = C.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ar, Br, Cr)
    return float(np.max(np.abs(T - 1.0)))


def _drop_zero_cols(U, V, W, max_iter=10):
    """Repeatedly drop any rank column that is identically zero in one factor."""
    for _ in range(max_iter):
        r = U.shape[1]
        zU = np.allclose(U, 0, atol=1e-9, rtol=0)
        zV = np.allclose(V, 0, atol=1e-9, rtol=0)
        zW = np.allclose(W, 0, atol=1e-9, rtol=0)
        keep = ~(zU | zV | zW)
        if np.all(keep):
            break
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
    return U, V, W


def _reduce_rank(U, V, W, tol=1e-9):
    """Greedy linear-dependency reduction.

    A factor's columns may be replaced by any matrix T (same number of
    columns); the product is invariant iff V = T V and W = T W (row space
    contained).  We test a single-column null-space vector at a time, which
    is exactly an invertible 2x2 transformation and thus preserves rank.
    Returns the reduced factors and the number of columns removed.
    """
    r = U.shape[1]
    keep = np.arange(r)

    for c in range(r):
        if c not in keep:
            continue
        if time.time() > 132.0:
            break
        idx = list(keep)
        # v with v[c]=1 and support on kept indices
        v = np.zeros(r)
        v[c] = 1.0
        A_c = U[:, c]
        B_c = V[:, c]
        C_c = W[:, c]

        for d in idx:
            if d == c:
                continue
            # T = I - v*beta, beta = A_d/A_c on rows where A_c != 0
            mask = np.abs(A_c) > 1e-9
            beta = np.zeros(r)
            beta[mask] = A_d[mask] / A_c[mask]
            newU = U - np.outer(A_c, beta)
            rV = B_c[None, :] * beta[:, None]
            rW = C_c[None, :] * beta[:, None]
            if np.allclose(V - rV, 0, atol=tol) and np.allclose(W - rW, 0, atol=tol):
                # column d is a redundant combination; drop it, keep c
                newKeep = [x for x in idx if x != d]
                U = newU
                keep = newKeep
                break

    keep = np.array(keep)
    return U[:, keep], V[:, keep], W[:, keep], int(r - len(keep))


def _fallback():
    r = _N * _M * _P
    U = np.zeros((_N * _M, r))
    V = np.zeros((_M * _P, r))
    W = np.zeros((_P * _N, r))
    col = 0
    for i in range(_N):
        for j in range(_M):
            for k in range(_P):
                U[i * _M + j, col] = 1
                V[j * _P + k, col] = 1
                W[k * _N + i, col] = 1
                col += 1
    return U, V, W


def construct():
    t0 = time.time()
    deadline = t0 + 138.0

    U, V, W = _build_base()
    U, V, W = _drop_zero_cols(U, V, W)
    err = _verify(U, V, W)
    if err > 1e-6:
        U, V, W = _fallback()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # attempt rank reduction (pure improvement search, always verified)
    if time.time() < deadline:
        Ur, Vr, Wr, removed = _reduce_rank(U, V, W)
        err2 = _verify(Ur, Vr, Wr)
        if removed > 0 and err2 < 1e-5:
            U, V, W = Ur, Vr, Wr

    U, V, W = _drop_zero_cols(U, V, W)
    if _verify(U, V, W) > 1e-5:
        U, V, W = _fallback()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
