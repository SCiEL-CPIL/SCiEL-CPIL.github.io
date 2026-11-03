"""Matrix multiplication 2,3,6 rank-minimal decomposition via search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_DEADLINE = 130.0  # safety margin under 150s kill
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _tensor_entries(n, m, p):
    """Yield (i, j, k) where the multiplication tensor is 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                yield (i, j, k)

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
    r = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build contraction
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i * m + j]
                vc = V[j * p + k]
                wc = W[k * n + i]
                s = np.dot(uc * vc, wc)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _als_decompose(n, m, p, r, seed=0, iters=4000, tol=1e-10):
    """Alternating least squares to find U,V,W of rank r decomposing the tensor."""
    rng = np.random.RandomState(seed)
    # Reshape tensor into a 3-tensor T[i, j, k] = 1
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    # This is not a standard 3-mode unfolding because indices are permuted.
    # For each c, define vectors: a_c over (i,j), b_c over (j,k), cvec_c over (k,i).
    # We need sum_c a_c[i,j] * b_c[j,k] * cvec_c[k,i] = 1.
    # ALS: fix two, solve for third via least squares.
    U = rng.randn(n * m, r) * 0.1
    V = rng.randn(m * p, r) * 0.1
    W = rng.randn(p * n, r) * 0.1
    # Precompute index lists
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    for it in range(iters):
        if _remaining() < 5.0:
            break
        # Solve for U: for each (i,j), U[i,j,:] should satisfy
        # sum_c U[i,j,c] * (V[j,k,c]*W[k,i,c]) = 1 for all k
        # Build linear system per (i,j): rows over k, cols over c
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k] * W[k * n + i]
                b = np.ones(p)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = sol
        # Solve for V: for each (j,k), sum_c V[j,k,c]*(U[i,j,c]*W[k,i,c]) = 1 for all i
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j] * W[k * n + i]
                b = np.ones(n)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = sol
        # Solve for W: for each (k,i), sum_c W[k,i,c]*(U[i,j,c]*V[j,k,c]) = 1 for all j
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j] * V[j * p + k]
                b = np.ones(m)
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = sol
        # Check convergence
        err = 0.0
        for (i, j, k) in triples:
            s = np.dot(U[i * m + j] * V[j * p + k], W[k * n + i])
            err += (s - 1.0) ** 2
        if err < tol:
            return U, V, W, err
    return U, V, W, err

def _to_rational_matrix(M, denom=2):
    """Round to multiples of 1/denom."""
    return np.round(M * denom) / denom

def construct():
    n, m, p = DIMENSIONS
    # Start from a known-good schoolbook as fallback
    Us, Vs, Ws = _schoolbook(n, m, p)
    best = (Us.tolist(), Vs.tolist(), Ws.tolist(), n * m * p)

    # Try decreasing ranks
    # Lower bound: max(n*m, m*p, n*p) = max(6,18,12)=18
    for r in range(18, n * m * p + 1):
        if _remaining() < 8.0:
            break
        found = None
        for seed in range(0, 6):
            if _remaining() < 8.0:
                break
            U, V, W, err = _als_decompose(n, m, p, r, seed=seed, iters=3000)
            if err < 1e-8:
                # Try to rationalize
                for denom in [1, 2, 4, 8, 16]:
                    Ur = _to_rational_matrix(U, denom)
                    Vr = _to_rational_matrix(V, denom)
                    Wr = _to_rational_matrix(W, denom)
                    if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                        found = (Ur, Vr, Wr, r)
                        break
                if found is None:
                    # use the float solution rounded to nearest rational with larger denom
                    for denom in [32, 64, 128]:
                        Ur = _to_rational_matrix(U, denom)
                        Vr = _to_rational_matrix(V, denom)
                        Wr = _to_rational_matrix(W, denom)
                        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                            found = (Ur, Vr, Wr, r)
                            break
                if found is not None:
                    break
        if found is not None:
            best = found
            break  # lowest r found

    U, V, W, rank = best
    return {"U": U, "V": V, "W": W}
