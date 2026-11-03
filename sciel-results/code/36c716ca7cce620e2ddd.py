import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition
    def get_schoolbook():
        R = rank_schoolbook
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = get_schoolbook()

    start_time = time.time()
    deadline = 140  # 10s safety margin

    def verify(U, V, W, atol=1e-6):
        R = U.shape[1]
        T_approx = np.zeros_like(T)
        for alpha in range(R):
            u = U[:, alpha]
            v = V[:, alpha]
            w = W[:, alpha]
            T_approx += np.einsum('i,j,k->ijk', u, v, w)
        return np.allclose(T_approx, T, atol=atol)

    # Try to find a lower rank decomposition using a structured approach
    # We'll try to exploit the structure of the tensor by using a "block" approach
    # where we group terms by shared indices.

    # For 3x4x5, let's try rank 55, 50, 45, etc.
    # We'll use a random restart approach with local optimization.

    def try_rank(R, max_iters=5000):
        """Try to find a rank-R decomposition using gradient descent on the tensor norm."""
        for restart in range(20):
            if time.time() - start_time > deadline:
                return None

            # Initialize with random vectors
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1

            # Scale to match the magnitude of the target tensor
            # The target tensor has 60 entries of 1, so the Frobenius norm is sqrt(60)
            # Each rank-1 term contributes roughly 1/60 to the norm if uniformly distributed
            scale = np.sqrt(60.0 / R)
            U *= scale
            V *= scale
            W *= scale

            # Gradient descent
            lr = 0.01
            for it in range(max_iters):
                if time.time() - start_time > deadline:
                    return None

                # Compute current approximation
                T_approx = np.zeros_like(T)
                for alpha in range(R):
                    T_approx += np.einsum('i,j,k->ijk', U[:, alpha], V[:, alpha], W[:, alpha])

                # Compute error
                E = T - T_approx
                err = np.linalg.norm(E)

                if err < 1e-6:
                    return U, V, W

                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for alpha in range(R):
                    v = V[:, alpha]
                    w = W[:, alpha]
                    dU[:, alpha] = np.einsum('ijk,j,k->i', E, v, w)
                    dV[:, alpha] = np.einsum('ijk,i,k->j', E, U[:, alpha], w)
                    dW[:, alpha] = np.einsum('ijk,i,j->k', E, U[:, alpha], v)

                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                # Adaptive learning rate
                if it % 100 == 0 and err > 1e-3:
                    lr *= 0.9

            # Check final result
            if verify(U, V, W):
                return U, V, W

        return None

    # Try ranks from 55 down to 40
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    for R in range(55, 39, -1):
        if time.time() - start_time > deadline:
            break
        result = try_rank(R, max_iters=2000)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, atol=1e-5):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break

    # If no improvement found, return schoolbook
    if best_rank == rank_schoolbook:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
