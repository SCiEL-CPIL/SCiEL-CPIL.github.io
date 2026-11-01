"""Search for rank-minimal exact tensor decomposition of 2x3 * 3x8 matmul."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _build_target_tensor(n, m, p):
    # T[i, j, k] = 1  (matmul tensor: C[i,k] = sum_j A[i,j] B[j,k])
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Return True if sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k (within tol)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok


def _verify_exact_fraction(U, V, W, n, m, p):
    """Exact rational verification. Entries must be Fraction or int."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    a = U[i * m + j][r]
                    b = V[j * p + k][r]
                    c = W[k * n + i][r]
                    if not isinstance(a, Fraction):
                        a = Fraction(a)
                    if not isinstance(b, Fraction):
                        b = Fraction(b)
                    if not isinstance(c, Fraction):
                        c = Fraction(c)
                    s += a * b * c
                if s != 1:
                    return False
    return True


def _snap_to_rational(matrix, max_den=8):
    """Snap each entry to nearest fraction with small denominator. Return list of lists of Fraction, or None if any entry far from rational."""
    out = []
    for row in matrix:
        newrow = []
        for v in row:
            best = None
            best_err = 1e-18
            for d in range(1, max_den + 1):
                num = round(v * d)
                frac = Fraction(num, d)
                err = abs(float(frac) - v)
                if err < best_err:
                    best_err = err
                    best = frac
            if best_err > 1e-6:
                return None
            newrow.append(best)
        out.append(newrow)
    return out


def _als_step(U, V, W, n, m, p, rank):
    """One ALS iteration: fix two, solve for the third via linear system. Returns new U,V,W."""
    U = np.array(U, dtype=float, copy=True)
    V = np.array(V, dtype=float, copy=True)
    W = np.array(W, dtype=float, copy=True)
    # Precompute targets: for each (i,j) solve for U[:, i*m+j] using V,W
    # U[i*m+j, r] = sum_k T[i,j,k] V[j*p+k, r] W[k*n+i, r]  (T=1)
    for i in range(n):
        for j in range(m):
            # Build A (rank x rank): A[r1,r2] = sum_k V[j*p+k, r1] W[k*n+i, r2]
            A = np.zeros((rank, rank))
            b = np.zeros(rank)
            for k in range(p):
                vcol = V[j * p + k, :]  # (rank,)
                wcol = W[k * n + i, :]  # (rank,)
                A += np.outer(vcol, wcol)
                b += vcol  # T=1
            try:
                U[i * m + j, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                pass
    # Now solve V: for each (j,k): V[j*p+k, r] = sum_i U[i*m+j, r] W[k*n+i, r'] ... 
    # Actually V[j*p+k, r2] * sum_i U[i*m+j, r1] W[k*n+i, r1] = W[k*n+k? ] no.
    # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,k.
    # For fixed (j,k), unknown V[j*p+k, :]. For each i: sum_r (U[i*m+j, r] W[k*n+i, r]) * V[j*p+k, r] = 1.
    for j in range(m):
        for k in range(p):
            A = np.zeros((rank, rank))
            b = np.zeros(rank)
            for i in range(n):
                ucol = U[i * m + j, :]
                wcol = W[k * n + i, :]
                A += np.outer(wcol, ucol)
                b += wcol
            try:
                V[j * p + k, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                pass
    # Solve W: for each (k,i): W[k*n+i, r] = ... 
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all j.
    for k in range(p):
        for i in range(n):
            A = np.zeros((rank, rank))
            b = np.zeros(rank)
            for j in range(m):
                ucol = U[i * m + j, :]
                vcol = V[j * p + k, :]
                A += np.outer(vcol, ucol)
                b += vcol
            try:
                W[k * n + i, :] = np.linalg.solve(A + 1e-12 * np.eye(rank), b)
            except np.linalg.LinAlgError:
                pass
    return U, V, W


def _error(U, V, W, n, m, p):
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err += (s - 1.0) ** 2
    return err


def _try_rank(rank, n, m, p, seed, max_iters=60, deadline=None):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    best_err = _error(U, V, W, n, m, p)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        U, V, W = _als_step(U, V, W, n, m, p, rank)
        err = _error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
        if best_err < 1e-10:
            break
    if best_err < 1e-6:
        # Try snapping to rationals
        for max_den in [2, 4, 8, 16]:
            Uf = _snap_to_rational(U, max_den)
            Vf = _snap_to_rational(V, max_den)
            Wf = _snap_to_rational(W, max_den)
            if Uf is None or Vf is None or Wf is None:
                continue
            if _verify_exact_fraction(Uf, Vf, Wf, n, m, p):
                return {"U": Uf, "V": Vf, "W": Wf}
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0  # safety margin before 150s kill
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_school

    # Try decreasing ranks with multiple seeds
    for rank in range(10, rank_school):
        if time.time() > deadline - 5:
            break
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            res = _try_rank(rank, n, m, p, seed=seed, max_iters=40, deadline=deadline - 5)
            if res is not None:
                best = res
                best_rank = rank
                break  # found exact at this rank, move to lower
    return best
# EVOLVE-BLOCK-END
