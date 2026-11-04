import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 140.0

    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 44]
    for target_rank in target_ranks:
        if target_rank >= rank_sb:
            continue
        if time.time() > deadline - 5:
            break
        try:
            result = search_rank(n, m, p, target_rank, deadline)
            if result is not None:
                return result
        except Exception:
            pass

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def compute_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    return error

def compute_gradients(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = T - 1.0
    gU3 = np.einsum('ijk,jkr,kir->ijr', residual, V3, W3)
    gV3 = np.einsum('ijk,ijr,kir->jkr', residual, U3, W3)
    gW3 = np.einsum('ijk,ijr,jkr->kir', residual, U3, V3)
    return gU3.reshape(n*m, r), gV3.reshape(m*p, r), gW3.reshape(p*n, r)

def search_rank(n, m, p, rank, deadline):
    np.random.seed(42)
    max_restarts = 8

    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break

        U = np.random.uniform(-1, 1, size=(n * m, rank))
        V = np.random.uniform(-1, 1, size=(m * p, rank))
        W = np.random.uniform(-1, 1, size=(p * n, rank))

        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2

        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = compute_error(U, V, W, n, m, p)

        lr = 0.5
        max_iters = 300
        no_improve = 0

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            gU, gV, gW = compute_gradients(U, V, W, n, m, p)

            new_U = U - lr * gU
            new_V = V - lr * gV
            new_W = W - lr * gW

            new_U = np.round(new_U * 2) / 2
            new_V = np.round(new_V * 2) / 2
            new_W = np.round(new_W * 2) / 2

            new_error = compute_error(new_U, new_V, new_W, n, m, p)

            if new_error < best_error - 1e-12:
                best_error = new_error
                best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
                U, V, W = new_U, new_V, new_W
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    break
                lr *= 0.8
                if lr < 0.01:
                    break

        if best_error < 1e-10:
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            if compute_error(U_snap, V_snap, W_snap, n, m, p) < 1e-10:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    return None
