"""Matrix multiplication 2x3x6 - rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

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

    best_result = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def to_numpy(U, V, W):
        return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

    def compute_error(U, V, W):
        """Compute the squared error of the decomposition."""
        U_np, V_np, W_np = to_numpy(U, V, W)
        r = U_np.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i * m + j], V_np[j * p + k])
                    val = np.dot(val, W_np[k * n + i])
                    err += (val - 1.0) ** 2
        return err

    def random_search(rank, iterations=10000):
        """Try to find a rank-r decomposition via random search."""
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None

        for it in range(iterations):
            if time.time() > deadline - 5:
                break

            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)

            err = compute_error(U.tolist(), V.tolist(), W.tolist())

            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()

            if err < 1e-12:
                return best_U, best_V, best_W, err

        return best_U, best_V, best_W, best_err

    def local_search(U_init, V_init, W_init, rank, iterations=5000, step_size=0.1):
        """Local search to refine a starting point."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        best_err = compute_error(U.tolist(), V.tolist(), W.tolist())

        for it in range(iterations):
            if time.time() > deadline - 5:
                break

            # Try random perturbations
            for _ in range(3):
                U_new = U + np.random.randn(*U.shape) * step_size
                V_new = V + np.random.randn(*V.shape) * step_size
                W_new = W + np.random.randn(*W.shape) * step_size

                err = compute_error(U_new.tolist(), V_new.tolist(), W_new.tolist())

                if err < best_err:
                    best_err = err
                    U = U_new
                    V = V_new
                    W = W_new
                    step_size *= 0.95
                else:
                    step_size *= 1.05
                    step_size = min(step_size, 1.0)

            if best_err < 1e-12:
                break

        return U, V, W, best_err

    # Try ranks from 20 up to 35
    for rank in range(20, min(36, best_rank)):
        if time.time() > deadline - 10:
            break

        # Random search
        U, V, W, err = random_search(rank, iterations=5000)

        if err is not None and err < 1e-9:
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break

        # Local search from best random point
        if U is not None:
            U, V, W, err = local_search(U, V, W, rank, iterations=3000)
            if err < 1e-9:
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break

    # If we found something better, return it
    if best_rank < n * m * p:
        return best_result

    # Otherwise return schoolbook
    return best_result
# EVOLVE-BLOCK-END
