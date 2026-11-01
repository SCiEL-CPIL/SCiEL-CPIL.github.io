"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.
Strategy: random rank-r factor search with local refinement + exact integer/half-integer
rounding verification. Falls back to schoolbook rank 72 if no better is found in time.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 140.0  # safety margin before 150s kill


def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
            # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _round_to_half(x):
    # round to nearest half-integer (multiple of 0.5)
    return np.round(2.0 * x) / 2.0


def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U[i * m + j, :], V[j * p + k, :])
                rhs = W[k * n + i, :]
                if np.max(np.abs(lhs - rhs)) > tol:
                    return False
    return True


def _try_rank(r, n, m, p, iters=2000, seed=0):
    """Random search for rank-r decomposition. Returns (U,V,W) or None."""
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        # random factors
        U = rng.normal(size=(n * m, r))
        V = rng.normal(size=(m * p, r))
        # W is determined: W[k*n+i, c] = sum_j U[i*m+j,c]*V[j*p+k,c]
        # Build W from U,V
        W = np.zeros((p * n, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    W[k * n + i, :] += U[i * m + j, :] * V[j * p + k, :]
        # Now the decomposition is automatically exact (by construction).
        # We need U,V,W to have integer/half-integer entries. Round them.
        U_r = _round_to_half(U)
        V_r = _round_to_half(V)
        W_r = _round_to_half(W)
        # After rounding, check exactness with rounded values.
        if _validate(U_r, V_r, W_r, n, m, p):
            # Convert to lists of ints/half-ints
            def to_list(M):
                return [[(int(v) if v == int(v) else float(v)) for v in row] for row in M]
            return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    return None


def _structured_rank(r, n, m, p):
    """Try structured decompositions: e.g., split m dimension.
    For (n x m)(m x p), a rank n*p + something decomposition may exist.
    Try: U[i*m+j, c] = A[i, c] * B[j, c] style (rank-n * rank-m outer).
    Actually for matrix mult, a known decomposition: rank = n*p + m*p + n*m - ... 
    Let's try the 'block' approach: treat B as having m rows; 
    C[i,k] = sum_j A[i,j] B[j,k].
    A rank-n*p decomposition: for each (i,k), C[i,k] = <row_i(A), col_k(B)>.
    That's rank n*p if we can represent inner product with n*p terms... no.
    Try rank = n*p: U[i*m+j, c] where c indexes (i',k')? Not directly.
    Skip; rely on random search + schoolbook fallback.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()  # rank 72 fallback
    best_rank = n * m * p

    # Try decreasing ranks from a reasonable lower bound up to best_rank-1.
    # Lower bound: max(n*p, m*p, n*m) = max(24, 24, 9) = 24.
    lower = max(n * p, m * p, n * m)
    seed_counter = 0
    for r in range(lower, best_rank):
        if time.time() > DEADLINE:
            break
        # budget per rank
        iters = 1500
        res = _try_rank(r, n, m, p, iters=iters, seed=seed_counter)
        seed_counter += 1
        if res is not None:
            best = res
            best_rank = r
            break  # found minimal in this range; stop (we go ascending)
    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", res["U"][0].__len__() if res else None)
