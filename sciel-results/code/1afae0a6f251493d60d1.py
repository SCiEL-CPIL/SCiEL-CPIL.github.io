"""Program for matrix multiplication tensor decomposition (3x4x5)."""
import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    schoolbook_rank = n * m * p  # 60

    # Build the schoolbook decomposition
    U0 = np.zeros((dim_u, schoolbook_rank))
    V0 = np.zeros((dim_v, schoolbook_rank))
    W0 = np.zeros((dim_w, schoolbook_rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    # Helper to verify decomposition
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - target) > 1e-6:
                        return False
        return True

    if not verify(U0, V0, W0):
        raise ValueError("Schoolbook decomposition failed verification")

    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = schoolbook_rank

    start_time = time.time()
    time_limit = 140  # Keep margin for 150s limit

    # Try to find a lower rank decomposition using a structured search
    # Strategy: Try to find a decomposition of rank r < 60
    # We use a combination of:
    # 1. Random initialization with sparse entries
    # 2. Local search / refinement
    # 3. Verification of exactness

    def tensor_residual(U, V, W):
        """Compute the tensor residual for the decomposition."""
        r = U.shape[1]
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res = abs(val - 1.0)
                    if res > max_res:
                        max_res = res
        return max_res

    def try_random_decomposition(r, max_iters=100):
        """Try to find a rank-r decomposition via random initialization and local search."""
        if r >= best_rank:
            return None

        # Initialize with random sparse entries
        U = np.random.choice([-1, 0, 1], size=(dim_u, r))
        V = np.random.choice([-1, 0, 1], size=(dim_v, r))
        W = np.random.choice([-1, 0, 1], size=(dim_w, r))

        # Local search: try flipping individual entries
        best_res = tensor_residual(U, V, W)
        if best_res < 1e-6:
            return (U, V, W)

        for iteration in range(max_iters):
            if time.time() - start_time > time_limit:
                return None

            improved = False
            # Try flipping a random entry in U
            idx_u = (random.randint(0, dim_u-1), random.randint(0, r-1))
            old_val_u = U[idx_u]
            U[idx_u] = -old_val_u if old_val_u != 0 else 1
            res = tensor_residual(U, V, W)
            if res < best_res:
                best_res = res
                improved = True
            else:
                U[idx_u] = old_val_u

            # Try flipping a random entry in V
            idx_v = (random.randint(0, dim_v-1), random.randint(0, r-1))
            old_val_v = V[idx_v]
            V[idx_v] = -old_val_v if old_val_v != 0 else 1
            res = tensor_residual(V, U, W)  # Note: this is wrong, need to fix
            # Actually, let's just compute the full residual
            res = tensor_residual(U, V, W)
            if res < best_res:
                best_res = res
                improved = True
            else:
                V[idx_v] = old_val_v

            # Try flipping a random entry in W
            idx_w = (random.randint(0, dim_w-1), random.randint(0, r-1))
            old_val_w = W[idx_w]
            W[idx_w] = -old_val_w if old_val_w != 0 else 1
            res = tensor_residual(U, V, W)
            if res < best_res:
                best_res = res
                improved = True
            else:
                W[idx_w] = old_val_w

            if best_res < 1e-6:
                return (U, V, W)

            if not improved:
                # Reset to a new random initialization
                U = np.random.choice([-1, 0, 1], size=(dim_u, r))
                V = np.random.choice([-1, 0, 1], size=(dim_v, r))
                W = np.random.choice([-1, 0, 1], size=(dim_w, r))
                best_res = tensor_residual(U, V, W)
                if best_res < 1e-6:
                    return (U, V, W)

        return None

    # Try ranks from 59 down to 50
    for r in range(best_rank - 1, max(best_rank - 10, 10), -1):
        if time.time() - start_time > time_limit:
            break

        result = try_random_decomposition(r, max_iters=50)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = r
                print(f"Found rank {r} decomposition!")
                break  # Stop searching once we find a lower rank

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
