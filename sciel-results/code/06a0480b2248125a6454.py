import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin before the 120s kill

    n, m, p = DIMENSIONS

    # Schoolbook decomposition (rank = n*m*p = 112)
    def schoolbook():
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
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    # Verify a decomposition is exact
    def verify(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try to find a lower-rank decomposition using gradient descent
    def try_gradient_descent(rank, max_iter=20000, lr=0.05, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5

        for it in range(max_iter):
            if time.time() > deadline:
                return None

            # Compute error and gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = V[j * p + k, :] * W[k * n + i, :]
                        val = np.dot(U[i * m + j, :], prod)
                        err = val - 1.0
                        grad_U[i * m + j, :] += err * prod
                        grad_V[j * p + k, :] += err * U[i * m + j, :] * W[k * n + i, :]
                        grad_W[k * n + i, :] += err * U[i * m + j, :] * V[j * p + k, :]

            # Update with adaptive learning rate
            norm_g = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if norm_g < 1e-12:
                break
            eff_lr = lr / (1.0 + norm_g)
            U -= eff_lr * grad_U
            V -= eff_lr * grad_V
            W -= eff_lr * grad_W

            # Periodic check
            if it % 500 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                if err < 1e-10:
                    return U, V, W

        # Final check
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        if err < 1e-8:
            return U, V, W
        return None

    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    # Try ranks from 60 to 111
    for r in range(60, rank_sb):
        if time.time() > deadline:
            break
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            result = try_gradient_descent(r, max_iter=5000, lr=0.1, seed=seed)
            if result is not None:
                U_try, V_try, W_try = result
                # Convert to lists and verify
                U_list = [[float(x) for x in row] for row in U_try.tolist()]
                V_list = [[float(x) for x in row] for row in V_try.tolist()]
                W_list = [[float(x) for x in row] for row in W_try.tolist()]
                if verify(U_list, V_list, W_list, tol=1e-7):
                    best_U, best_V, best_W, best_rank = U_list, V_list, W_list, r
                    found = True
                    break
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
