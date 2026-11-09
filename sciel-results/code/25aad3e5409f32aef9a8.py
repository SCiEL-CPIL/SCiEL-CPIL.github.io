"""Tensor decomposition search for 2x5x5 matrix multiplication.

The schoolbook rank is 50.  We attempt a structured random search over
half-integer entries to find a lower-rank exact decomposition.  If none is
found within the time budget, we return the schoolbook solution.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s wall-clock limit

    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 50

    # Build the target tensor (flattened)
    # T[i, j, k] = 1  (matrix multiplication tensor)
    # Flattened indices:
    #   U index: i*m + j   (0..9)
    #   V index: j*p + k   (0..24)
    #   W index: k*n + i   (0..24)
    # Tensor shape: (n*m, m*p, p*n) = (10, 25, 25)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    def schoolbook():
        rank = rank_school
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def check_decomposition(U, V, W):
        """Check if U ⊗ V ⊗ W exactly equals T."""
        # T_reconstructed[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # We compute this as sum_c U[:,c] * V[:,c] * W[:,c] (outer products summed)
        # For efficiency, compute column by column
        T_rec = np.zeros_like(T, dtype=np.float64)
        for c in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:, c].astype(np.float64),
                                V[:, c].astype(np.float64),
                                W[:, c].astype(np.float64))
        return np.allclose(T_rec, T.astype(np.float64), atol=1e-9)

    # Try a few structured random searches for lower rank
    # We'll try ranks from 40 down, but only if time permits
    # Use random half-integer entries (multiples of 0.5)
    best_rank = rank_school
    best_U, best_V, best_W = schoolbook()

    # Quick check: try rank 49, 48, ... with random initialization
    # This is very unlikely to succeed but we try a few iterations
    for target_rank in [49, 48, 47, 46]:
        if time.time() > deadline - 20.0:
            break

        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 20.0:
                break

            # Random half-integer entries in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
            vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)

            for c in range(target_rank):
                # Each column gets a random pattern
                U[:, c] = random.choice(vals)
                V[:, c] = random.choice(vals)
                W[:, c] = random.choice(vals)

            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break

    # Convert to lists for return
    U_list = best_U.astype(np.int64).tolist() if np.allclose(best_U, np.round(best_U)) else best_U.tolist()
    V_list = best_V.astype(np.int64).tolist() if np.allclose(best_V, np.round(best_V)) else best_V.tolist()
    W_list = best_W.astype(np.int64).tolist() if np.allclose(best_W, np.round(best_W)) else best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
