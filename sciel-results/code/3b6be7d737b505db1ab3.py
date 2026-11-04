"""Rewritten program: search for a rank < 24 exact half-integer tensor decomposition of M_{2,3,4}."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
_SEED = 20250101

def _schoolbook():
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

def _target_tensor(n, m, p):
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _reconstruct(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                recon[i, j, k] = s
    return recon

def _error(U, V, W, n, m, p, rank):
    recon = _reconstruct(U, V, W, n, m, p, rank)
    T = _target_tensor(n, m, p)
    return float(np.max(np.abs(recon - T)))

def _is_half_integer(arr, tol=1e-7):
    a = np.array(arr, dtype=float)
    return np.allclose(a * 2.0, np.round(a * 2.0), atol=tol)

def _snap(arr):
    return np.round(np.array(arr, dtype=float) * 2.0) / 2.0

def _random_candidate(rank, n, m, p, rng):
    # restricted half-integer values
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for c in range(rank):
        U[:, c] = rng.choice(vals, size=n * m)
        V[:, c] = rng.choice(vals, size=m * p)
        W[:, c] = rng.choice(vals, size=p * n)
    return U, V, W

def _local_refine(U, V, W, n, m, p, rank, rng, max_iters=200, tol=1e-10):
    """Local search: randomly perturb one entry of one matrix and keep if error decreases."""
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.0]
    best_err = _error(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank)
    if best_err < tol:
        return U, V, W, best_err
    best = (U.copy(), V.copy(), W.copy())
    for _ in range(max_iters):
        mat = rng.randint(0, 3)
        if mat == 1:
            r = rng.randint(0, n * m - 1)
            c = rng.randint(0, rank - 1)
            new_val = rng.choice(vals)
            U[r, c] = new_val
            err = _error(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            else:
                U[r, c] = best[0][r, c]
        elif mat == 2:
            r = rng.randint(0, m * p - 1)
            c = rng.randint(0, rank - 1)
            new_val = rng.choice(vals)
            V[r, c] = new_val
            err = _error(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            else:
                V[r, c] = best[1][r, c]
        else:
            r = rng.randint(0, p * n - 1)
            c = rng.randint(0, rank - 1)
            new_val = rng.choice(vals)
            W[r, c] = new_val
            err = _error(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            else:
                W[r, c] = best[2][r, c]
        if best_err < tol:
            break
    return best[0], best[1], best[2], best_err

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s out of 150s wall-clock
    n, m, p = DIMENSIONS
    fallback = _schoolbook()

    rng = random.Random(_SEED)

    # Try ranks from 20 up to 23 (24 is schoolbook)
    for rank in range(20, 24):
        if time.time() > deadline - 5.0:
            break
        found = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            U, V, W = _random_candidate(rank, n, m, p, rng)
            err = _error(U.tolist(), V.tolist(), W.tolist(), n, m, p, rank)
            if err < 1e-7:
                found = True
                break
            # local refine
            U, V, W, err = _local_refine(U, V, W, n, m, p, rank, rng, max_iters=100)
            if err < 1e-7:
                found = True
                break

        if found:
            U_s = _snap(U)
            V_s = _snap(V)
            W_s = _snap(W)
            # verify after snapping
            err_final = _error(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, rank)
            if err_final < 1e-9 and _is_half_integer(U_s) and _is_half_integer(V_s) and _is_half_integer(W_s):
                return {
                    "U": U_s.tolist(),
                    "V": V_s.tolist(),
                    "W": W_s.tolist()
                }

    return fallback
