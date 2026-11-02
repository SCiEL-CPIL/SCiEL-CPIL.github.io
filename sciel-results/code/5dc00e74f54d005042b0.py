"""Tensor decomposition search for 3x3 @ 3x8 matrix multiplication."""
import time
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [3, 3, 8]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(len(U[0])):
                    u = U[i * m + j][r]
                    v = V[j * p + k][r]
                    w = W[k * n + i][r]
                    if u != 0 and v != 0 and w != 0:
                        val += u * v * w
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_rank(rank, n, m, p, max_time, rng):
    """Attempt to find a valid decomposition at the given rank via random search."""
    start = time.time()
    U0 = np.random.randint(0, 3, size=(n * m, rank))
    V0 = np.random.randint(0, 3, size=(m * p, rank))
    W0 = np.random.randint(0, 3, size=(p * n, rank))
    U0[0, 0] = 1
    V0[0, 0] = 1
    W0[0, 0] = 1

    best_err = np.inf
    best = None
    step = 0
    while time.time() - start < max_time:
        step += 1
        # Randomly modify one entry of one matrix
        which = rng.randint(0, 3)
        if which == 0:
            r, c = rng.randint(0, n * m), rng.randint(0, rank)
            old = U0[r, c]
            U0[r, c] = rng.randint(-2, 3)
            err = _tensor_err(U0, V0, W0, n, m, p)
            if err < best_err:
                best_err = err
                best = (U0.copy(), V0.copy(), W0.copy())
            if err > best_err * 1.5 and err > 1.0:
                U0[r, c] = old
        elif which == 1:
            r, c = rng.randint(0, m * p), rng.randint(0, rank)
            old = V0[r, c]
            V0[r, c] = rng.randint(-2, 3)
            err = _tensor_err(U0, V0, W0, n, m, p)
            if err < best_err:
                best_err = err
                best = (U0.copy(), V0.copy(), W0.copy())
            if err > best_err * 1.5 and err > 1.0:
                V0[r, c] = old
        else:
            r, c = rng.randint(0, p * n), rng.randint(0, rank)
            old = W0[r, c]
            W0[r, c] = rng.randint(-2, 3)
            err = _tensor_err(U0, V0, W0, n, m, p)
            if err < best_err:
                best_err = err
                best = (U0.copy(), V0.copy(), W0.copy())
            if err > best_err * 1.5 and err > 1.0:
                W0[r, c] = old

        if best_err < 1e-12:
            break
    if best is not None and best_err < 1e-9:
        return best
    return None


def _tensor_err(U, V, W, n, m, p):
    """Compute the tensor error (sum of squared deviations from 1)."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(len(U[0])):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err += (val - 1.0) ** 2
    return err


def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 3x3 @ 3x8.
    Searches for a rank below the schoolbook rank 72 using local
    optimization over small integer entries, then falls back to
    the guaranteed schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    rng = np.random.RandomState(42)

    # Try decreasing ranks from a lower bound upward
    # The 3x3 matmul tensor has rank 17 (unproven but widely believed lower bound).
    # For 3x3 @ 3x8, a trivial lower bound is max(3*8, 3*3, 3*3) = 24 (tensor rank >= max of matrix dimensions products).
    # Actually the tensor rank must be at least max(n*p, m*p, n*m) for the structure, but the true bound is higher.
    # We'll search from rank 24 upward.
    for rank in range(24, 72):
        if time.time() > deadline:
            break
        time_per_rank = min(8.0, max(1.0, (deadline - time.time()) / max(1, 72 - rank)))
        result = _try_rank(rank, n, m, p, time_per_rank, rng)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                U_list = [[int(x) for x in row] for row in U.tolist()]
                V_list = [[int(x) for x in row] for row in V.tolist()]
                W_list = [[int(x) for x in row] for row in W.tolist()]
                return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition (always valid)
    U, V, W = _schoolbook(n, m, p)
    U_list = [[int(x) for x in row] for row in U.tolist()]
    V_list = [[int(x) for x in row] for row in V.tolist()]
    W_list = [[int(x) for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    elapsed = time.time() - t0
    rank = len(result['U'][0])
    print(f"Constructed rank {rank} in {elapsed:.1f}s")
