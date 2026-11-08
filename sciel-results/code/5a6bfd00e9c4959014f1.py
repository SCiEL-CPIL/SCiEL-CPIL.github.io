import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return the 7 rank-1 terms of the 2x2x2 Strassen decomposition."""
    U = [
        {(0,0):1,(1,1):1}, {(1,0):1,(1,1):1}, {(0,0):1},
        {(1,1):1}, {(0,0):1,(0,1):1}, {(1,0):1,(0,0):-1},
        {(0,1):1,(1,1):-1},
    ]
    V = [
        {(0,0):1,(1,1):1}, {(0,0):1}, {(0,1):1,(1,1):-1},
        {(1,0):1,(0,0):-1}, {(1,1):1}, {(0,0):1,(0,1):1},
        {(1,0):1,(1,1):1},
    ]
    W = [
        {(0,0):1,(1,1):1}, {(1,0):1,(1,1):-1}, {(0,1):1,(1,1):1},
        {(0,0):1,(1,0):1}, {(0,0):-1,(0,1):1}, {(1,1):1},
        {(0,0):1},
    ]
    return U, V, W


def _build_rank42():
    n, m, p = 2, 4, 6
    U, V, W = _strassen_222()
    r = 6 * 7
    Ub = np.zeros((n*m, r), dtype=float)
    Vb = np.zeros((m*p, r), dtype=float)
    Wb = np.zeros((p*n, r), dtype=float)
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), c in U[s].items():
                    Ub[i_loc, a*2 + j_loc] += c
                for (j_loc, k_loc), c in V[s].items():
                    Vb[a*2 + j_loc, b*2 + k_loc] += c
                for (i_loc, k_loc), c in W[s].items():
                    Wb[b*2 + k_loc, i_loc] += c
                col += 1
    return Ub, Vb, Wb


def _build_tensor_226():
    """Build the 2x2x6 matrix-multiplication tensor T[i,j,k]=1 at (i*m+j, j*p+k, k*n+i)."""
    n, m, p = 2, 2, 6
    I, J, K = n*m, m*p, p*n
    T = np.zeros((I, J, K), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T, (I, J, K)


def _als_226(T, rank, max_iter, tol, rng, deadline):
    """Fast ALS for CP decomposition of T. Returns (U, V, W, error) or None on timeout."""
    I, J, K = T.shape
    # Precompute transposed views for fast reshaping
    T_i = T.reshape(I, J*K)
    T_j = T.transpose(1, 0, 2).reshape(J, I*K)
    T_k = T.transpose(2, 0, 1).reshape(K, I*J)

    U = rng.standard_normal((I, rank))
    V = rng.standard_normal((J, rank))
    W = rng.standard_normal((K, rank))

    best_err = np.inf
    best = (U.copy(), V.copy(), W.copy())

    for it in range(max_iter):
        if time.time() > deadline:
            return None

        # Update U: T_i = U * (V ⊙ W)^T  where (V⊙W)[jk, s] = V[j,s]*W[k,s]
        # Build A: (J*K) x rank, A[j*K+k, s] = V[j,s] * W[k,s]
        A = (V.reshape(J,1,rank) * W.reshape(1,K,rank)).reshape(J*K, rank)
        U = T_i @ np.linalg.lstsq(A, T_i, rcond=None)[0].T if False else T_i @ np.linalg.pinv(A, hermitian=False)

        # Update V: T_j = V * (U ⊙ W)^T
        B = (U.reshape(I,1,rank) * W.reshape(1,K,rank)).reshape(I*K, rank)
        V = T_j @ np.linalg.pinv(B, hermitian=False)

        # Update W: T_k = W * (U ⊙ V)^T
        C = (U.reshape(I,1,rank) * V.reshape(1,J,rank)).reshape(I*J, rank)
        W = T_k @ np.linalg.pinv(C, hermitian=False)

        # Compute residual norm efficiently
        # recon[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s]
        # Use einsum for speed
        recon = np.einsum('is,js,ks->ijk', U, V, W)
        err = np.linalg.norm(T - recon)

        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < tol:
            break

    return best[0], best[1], best[2], best_err


def _to_rational_matrix(M, tol=1e-10):
    """Convert a float matrix to exact rationals (integers or halves) if entries are close to such."""
    out = []
    for row in M:
        r = []
        for x in row:
            xi = int(round(x))
            if abs(x - xi) < tol:
                r.append(xi)
            else:
                xh = round(x * 2)
                if abs(x - xh / 2.0) < tol:
                    r.append(Fraction(xh, 2))
                else:
                    r.append(Fraction(x).limit_denominator(1000000))
        out.append(r)
    return out


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # safety margin for 120s wall clock

    # Always-available rank-42 baseline
    Ub, Vb, Wb = _build_rank42()
    base_result = {
        "U": _to_rational_matrix(Ub),
        "V": _to_rational_matrix(Vb),
        "W": _to_rational_matrix(Wb),
    }

    # Try to find a rank-20 decomposition of the 2x2x6 tensor.
    # If successful, build a rank-40 full solution by using it twice (j=0,1 and j=2,3).
    T, dims = _build_tensor_226()
    I, J, K = dims

    best_U, best_V, best_W, best_err = None, None, None, np.inf
    seeds = [1, 7, 13, 42, 99, 200, 314, 577, 888, 1234]
    max_iter = 60

    for seed in seeds:
        if time.time() > deadline - 5:
            break
        rng = np.random.default_rng(seed)
        try:
            res = _als_226(T, 20, max_iter, 1e-10, rng, deadline - 5)
        except Exception:
            continue
        if res is None:
            continue
        U, V, W, err = res
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U, V, W

    # If we got a very small error, attempt rational recovery and verification
    if best_err is not None and best_err < 1e-7 and time.time() < deadline - 2:
        # Round to nearest integers or halves
        U_r = _to_rational_matrix(best_U)
        V_r = _to_rational_matrix(best_V)
        W_r = _to_rational_matrix(best_W)

        # Verify exactness by reconstructing T with rationals
        def _recon(Ur, Vr, Wr):
            Iu, Ju, Ku = Ur[0].__len__() if False else (len(Ur), len(Ur[0]), len(Vr[0]))
            # shapes: Ur is I x r, Vr is J x r, Wr is K x r
            Iu = len(Ur)
            r = len(Ur[0])
            Ju = len(Vr)
            Ku = len(Wr)
            out = np.zeros((Iu, Ju, Ku), dtype=float)
            for s in range(r):
                u = np.array([float(x) for x in Ur[:, s]], dtype=float) if False else np.array([float(Ur[i][s]) for i in range(Iu)], dtype=float)
                v = np.array([float(Vr[j][s]) for j in range(Ju)], dtype=float)
                w = np.array([float(Wr[k][s]) for k in range(Ku)], dtype=float)
                out += np.einsum('i,j,k->ijk', u, v, w)
            return out

        recon = _recon(U_r, V_r, W_r)
        if np.linalg.norm(recon - T) < 1e-9:
            # Build full rank-40 solution
            rank_full = 40
            U_full = np.zeros((n*m, rank_full), dtype=object)
            V_full = np.zeros((m*p, rank_full), dtype=object)
            W_full = np.zeros((p*n, rank_full), dtype=object)

            def _fill_block(offset):
                for s in range(20):
                    for i in range(2):
                        for j in range(2):
                            U_full[i*4 + j, offset + s] = U_r[i*2 + j][s]
                    for j in range(2):
                        for k in range(6):
                            V_full[j*6 + k, offset + s] = V_r[j*6 + k][s]
                    for k in range(6):
                        for i in range(2):
                            W_full[k*2 + i, offset + s] = W_r[k*2 + i][s]

            _fill_block(0)
            _fill_block(20)

            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist(),
            }

    return base_result
# EVOLVE-BLOCK-END
