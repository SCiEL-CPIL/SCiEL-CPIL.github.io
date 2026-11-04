"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin

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

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]

        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)

        return max_err < tol

    def tensor_error(U, V, W):
        """Compute sum of squared errors for tensor decomposition."""
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        rank = U_3d.shape[2]

        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                    err += (s - 1.0) ** 2
        return err

    def gradient_descent(U, V, W, lr=0.01, steps=5000, tol=1e-9):
        """Perform gradient descent to minimize tensor error."""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)

        for step in range(steps):
            if time.time() > _deadline - 10:
                break

            # Compute gradients
            dU = np.zeros_like(U_3d)
            dV = np.zeros_like(V_3d)
            dW = np.zeros_like(W_3d)

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
                        err_term = s - 1.0
                        dU[i, j, :] += 2 * err_term * (V_3d[j, k, :] * W_3d[k, i, :])
                        dV[j, k, :] += 2 * err_term * (U_3d[i, j, :] * W_3d[k, i, :])
                        dW[k, i, :] += 2 * err_term * (U_3d[i, j, :] * V_3d[j, k, :])

            # Update
            U_3d -= lr * dU
            V_3d -= lr * dV
            W_3d -= lr * dW

            # Check convergence
            if step % 100 == 0:
                err = tensor_error(U_3d, V_3d, W_3d)
                if err < tol:
                    break

        U_3d = U_3d.reshape(n * m, rank)
        V_3d = V_3d.reshape(m * p, rank)
        W_3d = W_3d.reshape(p * n, rank)

        return U_3d, V_3d, W_3d

    def random_search(max_rank, num_trials=200):
        """Try to find a decomposition of rank max_rank using gradient descent."""
        np.random.seed(42)

        for trial in range(num_trials):
            if time.time() > _deadline - 15:
                break

            # Random initialization
            U = np.random.randn(n * m, max_rank) * 0.5
            V = np.random.randn(m * p, max_rank) * 0.5
            W = np.random.randn(p * n, max_rank) * 0.5

            # Run gradient descent
            U, V, W = gradient_descent(U, V, W, lr=0.02, steps=3000)

            # Check if we found a solution
            err = tensor_error(U, V, W)
            if err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()

        return None

    def structured_search(rank, num_trials=50):
        """
        Try structured initializations based on Kronecker-like structure.
        The 2x4 by 4x5 multiplication can be viewed as two independent 1x4 by 4x5 multiplications.
        For 1x4 by 4x5, the tensor has 20 entries. We try to find low-rank decompositions.
        """
        np.random.seed(42)

        for trial in range(num_trials):
            if time.time() > _deadline - 15:
                break

            # Initialize with structure: separate the two rows
            # U has shape (8, rank), V has shape (20, rank), W has shape (10, rank)
            # Row 0 of U corresponds to i=0, row 1 to i=1
            # We initialize each "half" independently then mix slightly

            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))

            # Initialize first half (i=0)
            for r in range(rank // 2):
                U[0:m, r] = np.random.randn(m) * 0.5
                V[:, r] = np.random.randn(m * p) * 0.5
                W[0::n, r] = np.random.randn(p) * 0.5  # k*n + 0

            # Initialize second half (i=1)
            for r in range(rank // 2, rank):
                U[m:2*m, r] = np.random.randn(m) * 0.5
                V[:, r] = np.random.randn(m * p) * 0.5
                W[1::n, r] = np.random.randn(p) * 0.5  # k*n + 1

            # Run gradient descent
            U, V, W = gradient_descent(U, V, W, lr=0.02, steps=3000)

            # Check if we found a solution
            err = tensor_error(U, V, W)
            if err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()

        return None

    # Try ranks from 39 down to 20
    for rank in range(39, 19, -1):
        if time.time() > _deadline - 30:
            break

        # First try structured search
        result = structured_search(rank, num_trials=30)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

        # Then try random search
        result = random_search(rank, num_trials=50)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}

    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
