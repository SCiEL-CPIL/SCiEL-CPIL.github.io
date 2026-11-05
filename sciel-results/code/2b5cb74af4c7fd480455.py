"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill (150s wall-clock mentioned, use 120s limit)

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _exact_decompose(n, m, p, rank, seed=None, max_iters=200, tol=1e-10):
    """
    Try to find an exact (integer/half-integer) rank-rank decomposition of the
    matrix multiplication tensor T[i,j,k] = delta[i,k] (where output C[i,k] = sum_j A[i,j]B[j,k]).
    T is indexed (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
    Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    """
    rng = np.random.RandomState(seed if seed is not None else 0)

    # Build target tensor as a 3D array T[i,j,k]
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i, j, k] = 1.0  # delta[i,k] is 1 for all j; wait, C[i,k]=sum_j A[i,j]B[j,k], so T[i,j,k]=1 for all i,j,k? No.
    # Correction: T[i,j,k] = 1 for all i,j,k because C[i,k] = sum_j A[i,j] B[j,k] means the tensor
    # element coupling (i,j) from A, (j,k) from B, and (k,i) in W is 1 for every (i,j,k).
    # Indeed the schoolbook uses U[i*m+j, c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 per (i,j,k).
    # So T[i,j,k] = 1 for all i,j,k.
    T = np.ones((n, m, p), dtype=np.float64)

    # Random initialization
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)

    # Normalize columns to avoid scale drift
    for c in range(rank):
        nU = np.linalg.norm(U[:, c])
        nV = np.linalg.norm(V[:, c])
        nW = np.linalg.norm(W[:, c])
        if nU > 1e-12 and nV > 1e-12 and nW > 1e-12:
            scale = (nU * nV * nW) ** (1.0 / 3.0)
            U[:, c] /= scale
            V[:, c] /= scale
            W[:, c] /= scale

    best_err = np.inf
    best_U, best_V, best_W = None, None, None

    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        # Alternate: fix two, solve for the third via least squares.
        # For each (i,j), the row of U[i*m+j, :] must satisfy:
        #   sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k] for all k
        # So for fixed (i,j): define M_{k,c} = sum? Actually the product V[j*p+k,c]*W[k*n+i,c] is
        # elementwise over c. For fixed (i,j) and varying k, the coefficient of U[i*m+j,c] is
        #   a_c(k) = V[j*p+k, c] * W[k*n+i, c]
        # We need sum_c U[i*m+j,c] * a_c(k) = T[i,j,k] for all k. This is a linear system in U[i*m+j,:].
        # Build A_mat of shape (p, rank): A_mat[k, c] = V[j*p+k, c] * W[k*n+i, c]; b[k]=T[i,j,k]=1.
        # Solve A_mat @ U[i*m+j,:] = b.

        # Similarly for V[j*p+k,:] with fixed (j,k): B_mat[i, c] = U[i*m+j, c] * W[k*n+i, c], b[i]=1
        # For W[k*n+i,:] with fixed (k,i): C_mat[j, c] = U[i*m+j, c] * V[j*p+k, c], b[j]=1

        for _ in range(3):
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    b = np.ones(p)
                    sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
                    U[i * m + j, :] = sol
            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            A_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    b = np.ones(n)
                    sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
                    V[j * p + k, :] = sol
            # Update W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            A_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    b = np.ones(m)
                    sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
                    W[k * n + i, :] = sol

        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break

    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err

def _snap_to_halfint(U, V, W):
    """Try to snap entries to nearest half-integer and verify exactness."""
    n, m, p = DIMENSIONS
    U = np.round(U * 2) / 2.0
    V = np.round(V * 2) / 2.0
    W = np.round(W * 2) / 2.0
    # Verify
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                err += (s - 1.0) ** 2
    if np.sqrt(err) < 1e-8:
        return U, V, W, True
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    # Known lower bounds / target ranks to try
    # Schoolbook rank = 24. Try lower ranks.
    candidate_ranks = [18, 19, 20, 21, 22, 23, 24]

    best_result = None
    best_rank = None

    # First, always have a fallback: schoolbook rank 24
    def schoolbook():
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

    fallback = schoolbook()

    for rank in candidate_ranks:
        if _time_left() < 5.0:
            break
        found = False
        # Multiple random restarts
        n_restarts = 30
        for rs in range(n_restarts):
            if _time_left() < 3.0:
                break
            seed = rs * 1000 + rank
            U, V, W, err = _exact_decompose(n, m, p, rank, seed=seed, max_iters=150, tol=1e-10)
            if U is not None:
                # Try to snap to half-integers
                Us, Vs, Ws, ok = _snap_to_halfint(U, V, W)
                if ok:
                    result = {
                        "U": [[float(x) for x in row] for row in Us],
                        "V": [[float(x) for x in row] for row in Vs],
                        "W": [[float(x) for x in row] for row in Ws],
                    }
                    # Verify once more
                    if _verify(result):
                        best_result = result
                        best_rank = rank
                        found = True
                        break
                else:
                    # Maybe it's exact as-is (integers)?
                    result = {
                        "U": [[float(x) for x in row] for row in U],
                        "V": [[float(x) for x in row] for row in V],
                        "W": [[float(x) for x in row] for row in W],
                    }
                    if _verify(result):
                        best_result = result
                        best_rank = rank
                        found = True
                        break
        if found:
            break

    if best_result is not None:
        return best_result
    return fallback

def _verify(result):
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True
