"""Find a low-rank tensor decomposition for 4x5 x 5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

def _build_tensor():
    return np.ones((n, m, p))

def _compute_T(U, V, W):
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u[i, j] * v[j, :] * w[:, i]
    return T

def _gradients(U, V, W, residual):
    rank = U.shape[1]
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = residual[i, j, k]
                    dU[i * m + j, c] += r * v[j * p + k, c] * w[k * n + i, c]
                    dV[j * p + k, c] += r * u[i * m + j, c] * w[k * n + i, c]
                    dW[k * n + i, c] += r * u[i * m + j, c] * v[j * p + k, c]
    return dU, dV, dW

def _optimize(target_rank, max_iter=1000, lr=0.01, seed=42, deadline=None):
    np.random.seed(seed)
    U = np.random.randn(n * m, target_rank) * 0.3
    V = np.random.randn(m * p, target_rank) * 0.3
    W = np.random.randn(p * n, target_rank) * 0.3
    T_target = _build_tensor()
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        T_recon = _compute_T(U, V, W)
        residual = T_recon - T_target
        loss = np.sum(residual**2)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if loss < 1e-10:
            break
        dU, dV, dW = _gradients(U, V, W, residual)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    return best_U, best_V, best_W, best_loss

def _schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    for target_rank in [100, 90, 80, 70]:
        if time.time() > deadline - 15:
            break
        for seed in [42, 7]:
            if time.time() > deadline - 15:
                break
            U2, V2, W2, loss = _optimize(target_rank, max_iter=800, lr=0.02, seed=seed, deadline=deadline - 15)
            if loss < 1e-8 and target_rank < best_rank:
                best_rank = target_rank
                best_U, best_V, best_W = U2, V2, W2
                break
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    return {"U": U_list, "V": V_list, "W": W_list}
