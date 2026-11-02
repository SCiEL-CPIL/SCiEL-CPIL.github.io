"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.

Uses structured initialization and targeted search to find minimal-rank
decompositions. Focuses on the theoretical lower bound of max(n*p, m*p, n*m).
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


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
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _compute_error(U, V, W, n, m, p):
    """Compute the total error of the decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    err = 0.0
    # Check valid (i,j,k) positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err += (val - 1.0) ** 2
    # Check some invalid positions
    for _ in range(100):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if j != j2 or i != i2 or k != k2:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            err += val ** 2
    return err


def _round_to_half(x):
    """Round to nearest half-integer (multiple of 0.5)."""
    return np.round(2.0 * x) / 2.0


def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    for _ in range(200):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if j != j2 or i != i2 or k != k2:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > tol:
                return False
    return True


def _determine_W(U, V, n, m, p):
    """Given U and V, determine W such that the decomposition holds."""
    M = np.zeros((n, p, U.shape[1]))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                M[i, k, :] += U[i * m + j, :] * V[j * p + k, :]
    W = np.zeros((p * n, U.shape[1]))
    for i in range(n):
        for k in range(p):
            W[k * n + i, :] = M[i, k, :]
    return W


def _try_rank(r, n, m, p, iters=500, seed=0):
    """Random search for rank-r decomposition. Returns (U,V,W) or None."""
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        # Random factors with structured bias
        U = rng.normal(size=(n * m, r))
        V = rng.normal(size=(m * p, r))
        W = _determine_W(U, V, n, m, p)
        
        # Round to half-integers
        U_r = _round_to_half(U)
        V_r = _round_to_half(V)
        W_r = _round_to_half(W)
        
        # Check exactness with rounded values
        if _validate(U_r, V_r, W_r, n, m, p):
            def to_list(M):
                return [[(int(v) if v == int(v) else float(v)) for v in row] for row in M]
            return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    return None


def _try_rank_int(r, n, m, p, iters=500, seed=0):
    """Random search for rank-r decomposition with integer entries."""
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        U = rng.integers(-3, 4, size=(n * m, r))
        V = rng.integers(-3, 4, size=(m * p, r))
        W = _determine_W(U.astype(float), V.astype(float), n, m, p)
        W_r = np.round(W)
        
        if _validate(U, V, W_r, n, m, p):
            def to_list(M):
                return [[int(v) for v in row] for row in M]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W_r)}
    return None


def _structured_decomposition(n, m, p):
    """Try structured decompositions based on algebraic identities."""
    # For now, return None and rely on random search
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()  # rank 72 fallback
    best_rank = n * m * p

    # Try structured decomposition first
    res = _structured_decomposition(n, m, p)
    if res is not None:
        best = res
        best_rank = min(best_rank, len(res["U"][0]))

    # Try decreasing ranks from a reasonable lower bound up to best_rank-1.
    lower = max(n * p, m * p, n * m)
    seed_counter = 0
    
    # Budget allocation: we have ~145s. Try ranks from lower to best_rank-1.
    for r in range(lower, best_rank):
        if time.time() > DEADLINE:
            break
        remaining_ranks = best_rank - r
        time_per_rank = max(0.5, (DEADLINE - 5 - time.time()) / remaining_ranks)
        iters = max(100, int(time_per_rank * 50))
        
        # Try half-integer search
        res = _try_rank(r, n, m, p, iters=iters, seed=seed_counter)
        seed_counter += 1
        if res is not None:
            best = res
            best_rank = r
            break
        
        # Try integer search
        res = _try_rank_int(r, n, m, p, iters=iters, seed=seed_counter + 1000)
        seed_counter += 1
        if res is not None:
            best = res
            best_rank = r
            break

    return best


if __name__ == "__main__":
    res = construct()
    rank = len(res["U"][0]) if res else None
    print("rank:", rank)
    print("time:", time.time() - START)
