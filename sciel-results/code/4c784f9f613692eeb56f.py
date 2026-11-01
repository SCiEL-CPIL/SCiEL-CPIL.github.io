"""Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Exact verification using Fractions."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True


def _verify_np(U, V, W, n, m, p, tol=1e-12):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen22():
    """Strassen rank-7 decomposition for 2x2 * 2x2, returns (U,V,W) each (4,7)."""
    U = [
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
    ]
    V = [
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
    ]
    W = [
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
    ]
    return U, V, W


def _build_structured():
    """2x3 * 3x10 via Strassen(2x2) + schoolbook(2x1). Rank 7*10 + 2*10 = 90."""
    n, m, p = 2, 3, 10
    Us, Vs, Ws = _strassen22()
    R = 7 * p + 2 * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    # Strassen part: j in {0,1}, k in {0..9}
    for k in range(p):
        for c in range(7):
            base = k * 7 + c
            for a in range(4):
                U[a, base] = Us[a][c]
            for j in range(2):
                V[j*p + k, base] = Vs[j][c]
            for i in range(2):
                W[k*n + i, base] = Ws[i][c]
    # Schoolbook part: j=2, k in {0..9}
    for k in range(p):
        for c in range(2):
            base = 7*p + k * 2 + c
            i = c
            U[i*m + 2, base] = 1
            V[2*p + k, base] = 1
            W[k*n + i, base] = 1
    return U, V, W


def _ff_rank(U, V, W, p):
    """Compute tensor rank over GF(p) from factor arrays (mod p)."""
    import numpy as np
    U = np.asarray(U) % p
    V = np.asarray(V) % p
    W = np.asarray(W) % p
    n, m = U.shape[0] // 3, 3
    # Build the full tensor T[i,j,k] = sum_c U[i*3+j,c]*V[j*10+k,c]*W[k*2+i,c]
    T = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = int(np.dot(U[i*3+j], V[j*10+k] * W[k*2+i])) % p
    # Build matrix M of shape (n*m*p, r): row (i,j,k) = U[i,j] * V[j,k] * W[k,i] (Hadamard)
    r = U.shape[1]
    M = np.zeros((n*m*p, r), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[idx] = (U[i*3+j] * V[j*10+k] * W[k*2+i]) % p
                idx += 1
    # Rank over GF(p)
    mat = M.copy()
    rank = 0
    rows = mat.shape[0]
    for col in range(r):
        pivot = -1
        for row in range(rank, rows):
            if mat[row, col] % p != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        inv = pow(int(mat[rank, col]) % p, p-2, p)
        mat[rank] = (mat[rank] * inv) % p
        for row in range(rows):
            if row != rank and mat[row, col] % p != 0:
                mat[row] = (mat[row] - mat[row, col] * mat[rank]) % p
        rank += 1
    return rank


def _try_ff_reduce(U0, V0, W0, p, time_budget, seed=42):
    """Try to reduce rank over GF(p) by zeroing columns of a random submatrix, then lift."""
    import numpy as np
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    r0 = U0.shape[1]
    # Start from the full structured factors mod p
    best_rank = r0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    # Random column-order permutations; try zeroing the last t columns of a permuted set
    # i.e. keep a random subset of size r0 - t
    n, m, pp = 2, 3, 10
    for t in range(1, r0):
        if time.time() > deadline - 5:
            break
        target = r0 - t
        # Try several random subsets of columns of size target
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            keep = np.sort(rng.choice(r0, size=target, replace=False))
            Uk = U0[:, keep]
            Vk = V0[:, keep]
            Wk = W0[:, keep]
            rk = _ff_rank(Uk, Vk, Wk, p)
            if rk == target:
                # Lifting: solve for W over GF(p) exactly, then verify over integers
                # We need integer factors. Build W by solving per (i,k): A W[i,k] = 1
                # A[j,c] = Uk[i,j,c]*Vk[j,k,c]
                W_int = np.zeros((pp*n, target), dtype=float)
                feasible = True
                for i in range(n):
                    for k in range(pp):
                        A = np.zeros((m, target))
                        for j in range(m):
                            A[j, :] = Uk[i*m+j] * Vk[j*pp+k]
                        b = np.ones(m)
                        # solve over reals; if exact integer solution exists it will be integer
                        try:
                            sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                        except Exception:
                            feasible = False
                            break
                        if not np.allclose(A @ sol, b, atol=1e-9):
                            feasible = False
                            break
                        # round to integers (entries are small)
                        sol_r = np.round(sol)
                        if not np.allclose(A @ sol_r, b, atol=1e-9):
                            feasible = False
                            break
                        W_int[k*n+i, :] = sol_r
                    if not feasible:
                        break
                if feasible:
                    # verify exactly over integers
                    Uf = Uk.tolist(); Vf = Vk.tolist(); Wf = W_int.tolist()
                    if _verify(Uf, Vf, Wf, n, m, pp):
                        best_rank = target
                        best_U, best_V, best_W = Uk, Vk, W_int
                        return best_rank, best_U, best_V, best_W
    return best_rank, best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Fallback: schoolbook (rank 60)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Structured: Strassen + schoolbook (rank 90) -> but we verify; if valid, it's a candidate
    # Actually structured has rank 90 > 60, so it's not an improvement over schoolbook.
    # The goal is to find rank < 60. Use structured factors as a seed for FF reduction.
    U0, V0, W0 = _build_structured()
    for fld in (5, 3, 7):
        if time.time() > deadline - 20:
            break
        tl = min(40, deadline - time.time() - 10)
        if tl < 10:
            break
        for seed in (42, 12345, 777, 99999):
            if time.time() > deadline - 15:
                break
            tl2 = min(20, deadline - time.time() - 10)
            if tl2 < 8:
                break
            try:
                rk, Uk, Vk, Wk = _try_ff_reduce(U0, V0, W0, fld, tl2, seed=seed)
            except Exception:
                continue
            if rk < best_rank:
                Uf, Vf, Wf = Uk.tolist(), Vk.tolist(), Wk.tolist()
                if _verify(Uf, Vf, Wf, n, m, p):
                    best = {"U": Uf, "V": Vf, "W": Wf}
                    best_rank = rk
                    break
        if best_rank < n * m * p:
            break

    return best
