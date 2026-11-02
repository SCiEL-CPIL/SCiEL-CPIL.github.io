"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 142.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _target(n, m, p):
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    T[i, j, k] = 1.0
    return T

def _tensor_from(U, V, W, n, m, p, r):
    Ua = U.reshape(n, m, r)
    Va = V.reshape(m, p, r)
    Wa = W.reshape(p, n, r)
    return np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)

def _error(U, V, W, T, n, m, p, r):
    R = _tensor_from(U, V, W, n, m, p, r)
    return int(np.sum(np.abs(R - T) > 1e-9))

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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = _target(n, m, p)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    coeffs = [-2, -1, -0.5, 0, 0.5, 1, 2]
    rng = random.Random(777)

    def rand_init(r):
        U = np.array([[rng.choice(coeffs) for _ in range(r)] for _ in range(n*m)], dtype=float)
        V = np.array([[rng.choice(coeffs) for _ in range(r)] for _ in range(m*p)], dtype=float)
        W = np.array([[rng.choice(coeffs) for _ in range(r)] for _ in range(p*n)], dtype=float)
        return U, V, W

    def local_search(r, iters, seed):
        r_rng = random.Random(seed)
        U, V, W = rand_init(r)
        cur_err = _error(U, V, W, T, n, m, p, r)
        best_err = cur_err
        best_state = (U.copy(), V.copy(), W.copy())
        T0 = 1.0
        for it in range(iters):
            if _remaining() < 2:
                break
            which = r_rng.randint(0, 2)
            if which == 0:
                row = r_rng.randint(0, n*m-1); col = r_rng.randint(0, r-1)
                old = U[row, col]; U[row, col] = r_rng.choice(coeffs)
                new_err = _error(U, V, W, T, n, m, p, r)
                if new_err < cur_err or (new_err <= cur_err and r_rng.random() < 0.3):
                    cur_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_state = (U.copy(), V.copy(), W.copy())
                        if best_err == 0:
                            break
                else:
                    U[row, col] = old
            elif which == 1:
                row = r_rng.randint(0, m*p-1); col = r_rng.randint(0, r-1)
                old = V[row, col]; V[row, col] = r_rng.choice(coeffs)
                new_err = _error(U, V, W, T, n, m, p, r)
                if new_err < cur_err or (new_err <= cur_err and r_rng.random() < 0.3):
                    cur_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_state = (U.copy(), V.copy(), W.copy())
                        if best_err == 0:
                            break
                else:
                    V[row, col] = old
            else:
                row = r_rng.randint(0, p*n-1); col = r_rng.randint(0, r-1)
                old = W[row, col]; W[row, col] = r_rng.choice(coeffs)
                new_err = _error(U, V, W, T, n, m, p, r)
                if new_err < cur_err or (new_err <= cur_err and r_rng.random() < 0.3):
                    cur_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_state = (U.copy(), V.copy(), W.copy())
                        if best_err == 0:
                            break
                else:
                    W[row, col] = old
        return best_err, best_state

    # Try ranks below schoolbook, from low to high.
    for r in range(15, best_rank):
        if _remaining() < 6:
            break
        # a few restarts per rank
        for restart in range(4):
            if _remaining() < 4:
                break
            iters = max(2000, int(20000 * _remaining() / 30))
            err, state = local_search(r, iters, seed=1000 + r*10 + restart)
            if err == 0:
                U2, V2, W2 = state
                # verify
                if _error(U2, V2, W2, T, n, m, p, r) == 0:
                    best_rank = r
                    best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                    return best
    return best
# EVOLVE-BLOCK-END
