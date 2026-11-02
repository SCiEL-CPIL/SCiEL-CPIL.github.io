import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    # Schoolbook solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_result(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    best_rank = rank_school
    best_result = to_result(U_school, V_school, W_school)

    def check_exact(U, V, W, tol=1e-8):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - target)) < tol

    def try_rank(r, n_trials=5):
        nonlocal best_rank, best_result
        if r >= best_rank:
            return
        if time.time() > deadline - 10:
            return

        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break

            np.random.seed(trial * 98765 + 4321)
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5

            lr = 0.05
            best_loss = np.inf
            for step in range(100):
                if time.time() > deadline - 10:
                    break

                T = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - target
                loss = np.sum(diff ** 2)

                if loss < 1e-10:
                    if check_exact(U, V, W):
                        best_rank = r
                        best_result = to_result(U, V, W)
                        return
                    break

                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                dU = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
                dV = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
                dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V)

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

            if best_loss < 1e-10:
                if check_exact(best_U, best_V, best_W):
                    best_rank = r
                    best_result = to_result(best_U, best_V, best_W)
                    return

    for r in range(30, rank_school):
        try_rank(r)
        if best_rank <= r:
            break

    return best_result
