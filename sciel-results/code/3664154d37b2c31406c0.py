import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook fallback (rank 60)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try ranks from low to high
    for rank in range(15, 60):
        if time.time() > deadline - 15:
            break

        found = _try_rank_als(n, m, p, rank, deadline)
        if found is not None:
            U, V, W = found
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
            break

    return best_result


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank_als(n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition using alternating least squares."""
    rng = np.random.default_rng(42)
    n_restarts = 40
    max_iters = 500

    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break

        # Random initialization
        scale = 0.5
        U = rng.standard_normal((n * m, rank)) * scale
        V = rng.standard_normal((m * p, rank)) * scale
        W = rng.standard_normal((p * n, rank)) * scale

        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Update W: for each (i,k), solve linear system
            for i in range(n):
                for k in range(p):
                    A = U[i*m:(i+1)*m, :] * V[k*p:(k+1)*p, :]  # (m, rank)
                    b = np.ones(m)
                    W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]

            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]

            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]

            # Compute error
            err = _compute_error(n, m, p, U, V, W)

            # Check convergence
            if err < 1e-12:
                result = _rationalize(n, m, p, U, V, W)
                if result is not None:
                    return result

            # Early stop if no progress
            if it > 50 and abs(prev_err - err) < 1e-15 and err > 1e-6:
                break
            prev_err = err

    return None


def _compute_error(n, m, p, U, V, W):
    """Compute max absolute error over all (i,j,k) triples."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _rationalize(n, m, p, U, V, W):
    """Try to rationalize to integers or half-integers."""
    # Try half-integers
    U2 = np.round(2 * U) / 2
    V2 = np.round(2 * V) / 2
    W2 = np.round(2 * W) / 2
    if _compute_error(n, m, p, U2, V2, W2) < 1e-6:
        return (U2.tolist(), V2.tolist(), W2.tolist())

    # Try integers
    Ui = np.round(U)
    Vi = np.round(V)
    Wi = np.round(W)
    if _compute_error(n, m, p, Ui, Vi, Wi) < 1e-6:
        return (Ui.tolist(), Vi.tolist(), Wi.tolist())

    # Try quarter-integers as intermediate
    U4 = np.round(4 * U) / 4
    V4 = np.round(4 * V) / 4
    W4 = np.round(4 * W) / 4
    if _compute_error(n, m, p, U4, V4, W4) < 1e-6:
        return (U4.tolist(), V4.tolist(), W4.tolist())

    return None
