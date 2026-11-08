"""Search for a low-rank exact tensor decomposition of the 4x4 * 4x7 multiplication tensor."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

_START = time.time()
_DEADLINE = 150.0 - 20.0   # safety margin

def _slices():
    """Return the 112 multiplication-tensor slices, each a (16,28) matrix."""
    n, m, p = DIMENSIONS
    sl = []
    for i in range(n):
        for k in range(p):
            M = np.zeros((n * m, m * p))
            for j in range(m):
                M[i * m + j, j * p + k] = 1.0
            sl.append(M)
    return sl

def _flat(slices):
    return [s.reshape(-1) for s in slices]

def _span_check(flat, cols):
    """Return True if all slice vectors lie in span of cols (tolerance-based)."""
    if len(cols) == 0:
        return len(flat) == 0
    A = np.column_stack(cols)
    Q, _ = np.linalg.qr(A)
    for v in flat:
        resid = v - Q @ (Q.T @ v)
        if np.linalg.norm(resid) > 1e-6:
            return False
    return True

def _basis_from_sums(flat, cols, seed):
    """Build a candidate basis by random linear combinations of the slices."""
    rng = random.Random(seed)
    idxs = list(range(len(flat)))
    basis = []
    for _ in range(len(flat)):
        if not _span_check(flat, basis):
            basis.append(flat[rng.choice(idxs)])
        else:
            break
    return basis

def _try_rank(flat, r, rng, n_trials=200):
    """Try to find r basis tensors whose span covers all slices."""
    for _ in range(n_trials):
        if time.time() > _DEADLINE:
            return None
        # random linear combinations of r slices
        idxs = list(range(len(flat)))
        cols = []
        for _ in range(r):
            c = np.zeros(flat[0].shape)
            for _ in range(8):
                a = rng.uniform(-2.0, 2.0)
                c = c + a * flat[rng.choice(idxs)]
            if np.linalg.norm(c) > 1e-8:
                cols.append(c)
        if len(cols) < r:
            continue
        if _span_check(flat, cols):
            # reduce to a minimal spanning set of <= r
            basis = []
            for c in cols:
                if not _span_check(flat, basis):
                    basis.append(c)
                else:
                    break
            if len(basis) <= r and _span_check(flat, basis):
                return basis
    return None

def _build_result(basis, rank):
    n, m, p = DIMENSIONS
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c, mat in enumerate(basis):
        M = mat.reshape(n * m, m * p)
        for idx in range(n * m):
            for jdx in range(m * p):
                val = float(M[idx, jdx])
                if val == 0.0:
                    continue
                U[idx][c] = val
                V[jdx][c] = val
                # W row index k*n + i where idx=i*m+j, jdx=j*p+k
                i = idx // m
                j = idx % m
                jj = jdx // p
                k = jdx % p
                # only when j == jj (the product term)
                if j == jj:
                    W[k * n + i][c] = val
    return {"U": U, "V": V, "W": W}

def _verify(result, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    for i in range(n):
        for k in range(p):
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c])
            val = 0.0
            for c in range(U.shape[1]):
                inner = 0.0
                for j in range(m):
                    inner += U[i * m + j, c] * V[j * p + k, c]
                val += W[k * n + i, c] * inner
            if abs(val - 1.0) > tol:
                return False
    return True

def _schoolbook_basis():
    """Return the schoolbook (rank n*m*p) basis."""
    n, m, p = DIMENSIONS
    school = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(n * m * m * p)
                vec[(i * m + j) * (m * p) + (j * p + k)] = 1.0
                school.append(vec)
    return school

def construct():
    global _START
    n, m, p = DIMENSIONS
    flat = _flat(_slices())
    rng = random.Random(12345)

    # Start with schoolbook rank as a safe fallback
    best_rank = n * m * p
    best_basis = _schoolbook_basis()

    # Search for lower ranks
    # The tensor rank for (n,m,p) matrix multiplication is at most n*m*p
    # and at least max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Try ranks from 28 upward
    for r in range(28, best_rank + 1):
        if time.time() > _DEADLINE:
            break
        basis = _try_rank(flat, r, rng, n_trials=300)
        if basis is not None:
            # Verify the result
            result = _build_result(basis, len(basis))
            if _verify(result):
                best_rank = len(basis)
                best_basis = basis
                break

    if best_basis is None:
        best_basis = _schoolbook_basis()

    result = _build_result(best_basis, len(best_basis))
    return result
