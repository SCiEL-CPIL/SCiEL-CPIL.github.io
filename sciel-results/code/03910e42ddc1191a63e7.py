"""
Rank-minimal tensor decomposition for 4x4 by 4x7 matrix multiplication.

Strategy: build the multiplication tensor, get an initial low-rank factorization
via truncated SVD, refine with alternating least squares (ALS), then snap the
factor matrices to the nearest small rational (denominator <= 4) to obtain exact
integer/half-integer entries. Return the best (lowest-rank) candidate that
exactly reproduces the tensor.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]


def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[u, v, w].

    u = i*m + j  (A index: row i, col j)
    v = j*p + k  (B index: row j, col k)
    w = k*n + i  (C index: row k, col i)  [trace convention W[k*n+i, r]]
    T[u, v, w] = 1 iff the j in u matches the j in v.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _snap_rational(M, max_den=4):
    """Snap each entry of M to the nearest rational with denominator <= max_den."""
    out = np.zeros_like(M)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            x = M[a, b]
            if abs(x) < 1e-9:
                out[a, b] = 0.0
                continue
            best = (0, 0.0)
            for d in range(1, max_den + 1):
                num = round(x * d)
                val = num / d
                err = abs(x - val)
                if err < best[0]:
                    best = (err, val)
            out[a, b] = best[1]
    return out


def _verify(U, V, W, T, tol=1e-6):
    """Check whether sum_r U[:,r] V[:,r]^T W[:,r]^T == T (up to tolerance)."""
    # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    # Form via einsum
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(recon, T, atol=tol)


def _als_decompose(T, rank, n_iter=60, seed=0):
    """Alternating least squares to find rank-rank factors U,V,W of tensor T."""
    rng = np.random.default_rng(seed)
    nU, nV, nW = T.shape
    U = rng.standard_normal((nU, rank))
    V = rng.standard_normal((nV, rank))
    W = rng.standard_normal((nW, rank))

    # Precompute the mode-1, mode-2, mode-3 matricizations of T
    T1 = T.reshape(nU, nV * nW)          # rows: u
    T2 = T.reshape(nV, nU * nW)          # rows: v
    T3 = T.reshape(nW, nU * nV)          # rows: w

    from scipy.sparse.linalg import lsqr

    for it in range(n_iter):
        # Solve for U: for each u, T1[u,:] = sum_r U[u,r] * (V[:,r] kron W[:,r])
        # Design matrix D1[r, (v,w)] = V[v,r]*W[w,r]
        D1 = np.einsum('vr,wr->rvw', V, W).reshape(rank, nV * nW)
        # Solve D1^T X = T1^T  =>  X = U
        try:
            U = lsqr(D1.T, T1.T, atol=1e-9, btol=1e-9, iter_lim=50)[0]
        except Exception:
            break
        # Solve for V
        D2 = np.einsum('ur,wr->ruw', U, W).reshape(rank, nU * nW)
        try:
            V = lsqr(D2.T, T2.T, atol=1e-9, btol=1e-9, iter_lim=50)[0]
        except Exception:
            break
        # Solve for W
        D3 = np.einsum('ur,vr->rvu', U, V).reshape(rank, nU * nV)
        try:
            W = lsqr(D3.T, T3.T, atol=1e-9, btol=1e-9, iter_lim=50)[0]
        except Exception:
            break
    return U, V, W


def _svd_decompose(T, rank, seed=0):
    """Initial rank-approximation via truncated SVD of the mode-1 matricization."""
    rng = np.random.default_rng(seed)
    nU, nV, nW = T.shape
    # Random projection to break symmetry, then SVD on a flattened view
    T1 = T.reshape(nU, nV * nW)
    # Use randomized SVD via power iterations on T1
    Q = rng.standard_normal((nV * nW, rank + 4))
    Y = T1 @ Q
    Qy, _ = np.linalg.qr(Y)
    B = Qy.T @ T1
    U_s, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = T1 @ Qy @ U_s[:, :rank]
    V = Qy @ U_s[:, :rank] * s[:rank]
    # Now T1 ~= U @ V.T. But we need T1[u, (v,w)] = sum_r U[u,r]*M[v,w,r]
    # Let M = V.T reshaped to (nV, nW, rank)
    M = V.T.reshape(nV, nW, rank)
    # We need M[v,w,r] = V[v,r]*W[w,r]. Factor M via another SVD per r.
    Vm = np.zeros((nV, rank))
    Wm = np.zeros((nW, rank))
    for r in range(rank):
        Mr = M[:, :, r]  # nV x nW
        Ur, sr, Vtr = np.linalg.svd(Mr, full_matrices=False)
        Vm[:, r] = Ur[:, 0] * (sr[0] ** 0.5)
        Wm[:, r] = Vtr[0, :] * (sr[0] ** 0.5)
    return U, Vm, Wm


def _refine_and_snap(T, U0, V0, W0, max_den=4, n_iter=30, seed=0):
    """ALS refine then snap to rationals; verify."""
    U, V, W = _als_decompose(T, U0.shape[1], n_iter=n_iter, seed=seed)
    Us = _snap_rational(U, max_den)
    Vs = _snap_rational(V, max_den)
    Ws = _snap_rational(W, max_den)
    return Us, Vs, Ws


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130

    T = _build_tensor(n, m, p)

    # Schoolbook baseline (rank 112)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best = (rank_school, U_sb, V_sb, W_sb)

    # Try a range of ranks with SVD init + ALS + snapping
    for rank in [64, 72, 80, 88, 96, 104, 112]:
        if time.time() > deadline - 20:
            break
        for seed in [0, 1, 2]:
            if time.time() > deadline - 15:
                break
            try:
                U0, V0, W0 = _svd_decompose(T, rank, seed=seed)
                Us, Vs, Ws = _refine_and_snap(T, U0, V0, W0, max_den=4,
                                              n_iter=30, seed=seed)
                if _verify(Us, Vs, Ws, T, tol=1e-9):
                    if rank < best[0]:
                        best = (rank, Us, Vs, Ws)
            except Exception:
                continue

    rank_b, Ub, Vb, Wb = best
    U = [[float(x) for x in Ub[r]] for r in range(Ub.shape[0])]
    V = [[float(x) for x in Vb[r]] for r in range(Vb.shape[0])]
    W = [[float(x) for x in Wb[r]] for r in range(Wb.shape[0])]
    return {'U': U, 'V': V, 'W': W}
