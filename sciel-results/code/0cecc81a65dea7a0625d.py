"""Rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 120.0


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :] * V[j * p + k, :], W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _to_frac(M, tol=1e-9):
    """Convert a float matrix to exact fractions (half-integers)."""
    M = np.array(M, dtype=float)
    # Try to snap to half-integers
    M2 = np.round(M * 2) / 2
    if np.allclose(M, M2, atol=tol):
        M = M2
    out = []
    for row in M:
        r = []
        for x in row:
            f = Fraction(str(round(float(x), 12)))
            r.append(f)
        out.append(r)
    return out


def _als_decompose(n, m, p, rank, n_iter=200, seed=42):
    """Alternating least squares to find rank-r decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1

    # Precompute index mappings
    idx_U = [(i * m + j) for i in range(n) for j in range(m)]
    idx_V = [(j * p + k) for j in range(m) for k in range(p)]
    idx_W = [(k * n + i) for k in range(p) for i in range(n)]

    for it in range(n_iter):
        if time.time() > _DEADLINE - 10:
            break
        # Update U: for each (i,j), solve for U[i*m+j,:]
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                # Build system: sum_k W[k*n+i,:] * (V[j*p+k,:] * U[ui,:]) ... 
                # Actually: sum_c U[ui,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                # Let x = U[ui,:], then sum_c x[c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # This is: (V[j*p+k,:] * W[k*n+i,:]) dot x = 1 for each k
                # So A x = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c], b[k]=1
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                U[ui, :] = x

        # Update V: for each (j,k), solve for V[j*p+k,:]
        for j in range(m):
            for k in range(p):
                vj = j * p + k
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                V[vj, :] = x

        # Update W: for each (k,i), solve for W[k*n+i,:]
        for k in range(p):
            for i in range(n):
                wk = k * n + i
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                x, *_ = np.linalg.lstsq(A, b, rcond=None)
                W[wk, :] = x

        # Check convergence
        if it % 20 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j,:] * V[j*p+k,:], W[k*n+i,:])
                        err = max(err, abs(val - 1.0))
            if err < 1e-10:
                break

    # Final error check
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j,:] * V[j*p+k,:], W[k*n+i,:])
                err = max(err, abs(val - 1.0))
    return U, V, W, err


def _try_rank(n, m, p, rank, n_trials=5, n_iter=150):
    """Try to find a valid rank-r decomposition with multiple random inits."""
    best = None
    best_err = float('inf')
    for trial in range(n_trials):
        if time.time() > _DEADLINE - 10:
            break
        U, V, W, err = _als_decompose(n, m, p, rank, n_iter=n_iter, seed=trial*1000+42)
        if err < best_err:
            best_err = err
            best = (U, V, W)
        if err < 1e-8:
            return U, V, W, err
    return best[0], best[1], best[2], best_err if best else (None, None, None, float('inf'))


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    # Try ranks from low to high
    for rank in [20, 25, 30, 35, 40, 45]:
        if time.time() > _DEADLINE - 15:
            break
        U, V, W, err = _try_rank(n, m, p, rank, n_trials=3, n_iter=200)
        if err < 1e-8:
            # Verify with the proper check
            if _verify(U, V, W, n, m, p):
                Uf = _to_frac(U)
                Vf = _to_frac(V)
                Wf = _to_frac(W)
                return {"U": Uf, "V": Vf, "W": Wf}
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
