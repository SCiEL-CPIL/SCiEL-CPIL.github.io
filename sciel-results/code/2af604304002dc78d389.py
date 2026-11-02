"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 140

    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U_mat, V_mat, W_mat):
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        rank = U_arr.shape[1]
        recon = np.zeros_like(target)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        return np.allclose(recon, target, atol=1e-8)

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
        return {"U": U, "V": V, "W": W}

    def compute_loss(U, V, W, target):
        recon = np.zeros_like(target)
        for s in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
        return np.sum((target - recon)**2)

    def polish(U, V, W, target, max_iter=300):
        """Gradient-based polish."""
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            recon = np.zeros_like(target)
            for s in range(U.shape[1]):
                recon += np.einsum('a,b,c->abc', U[:,s], V[:,s], W[:,s])
            residual = target - recon
            loss = np.sum(residual**2)
            if loss < 1e-12:
                break
            grad_U = np.einsum('abc,bs,cs->as', residual, V, W)
            grad_V = np.einsum('abc,as,cs->bs', residual, U, W)
            grad_W = np.einsum('abc,as,bs->cs', residual, U, V)
            lr = 0.01 / (1 + it * 0.002)
            U = U - lr * grad_U
            V = V - lr * grad_V
            W = W - lr * grad_W
        return U, V, W

    def try_rank(r, n_restarts=20):
        """Try to find a rank-r decomposition using ALS + gradient polish."""
        if time.time() > deadline - 20:
            return None
        best_loss = np.inf
        best_U, best_V, best_W = None, None, None
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            np.random.seed(restart * 104729 + r * 7919)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            # ALS phase
            for it in range(80):
                if time.time() > deadline - 20:
                    break
                # Update U
                M_U = np.zeros((n*m*m*p*p*n, r))
                for s in range(r):
                    M_U[:,s] = (V[:,s] * W[:,s]).reshape(-1)
                try:
                    A = M_U.T @ M_U
                    b = M_U.T @ target.reshape(-1)
                    U = np.linalg.solve(A + 1e-8*np.eye(r), b)
                except:
                    pass
                # Update V
                M_V = np.zeros((n*m*m*p*p*n, r))
                for s in range(r):
                    M_V[:,s] = (U[:,s] * W[:,s]).reshape(-1)
                try:
                    A = M_V.T @ M_V
                    b = M_V.T @ target.reshape(-1)
                    V = np.linalg.solve(A + 1e-8*np.eye(r), b)
                except:
                    pass
                # Update W
                M_W = np.zeros((n*m*m*p*p*n, r))
                for s in range(r):
                    M_W[:,s] = (U[:,s] * V[:,s]).reshape(-1)
                try:
                    A = M_W.T @ M_W
                    b = M_W.T @ target.reshape(-1)
                    W = np.linalg.solve(A + 1e-8*np.eye(r), b)
                except:
                    pass
                loss = compute_loss(U, V, W, target)
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if loss < 1e-10:
                    break
            # Gradient polish if ALS made progress
            if best_loss < 5.0:
                U_p, V_p, W_p = polish(U, V, W, target, max_iter=150)
                loss_p = compute_loss(U_p, V_p, W_p, target)
                if loss_p < best_loss:
                    best_loss = loss_p
                    best_U, best_V, best_W = U_p, V_p, W_p
        if best_loss < 1e-8 and best_U is not None:
            # Try to round to simple rationals
            for scale in [1, 2, 3, 4, 5, 10, 20, 50, 100]:
                U_r = np.round(best_U * scale) / scale
                V_r = np.round(best_V * scale) / scale
                W_r = np.round(best_W * scale) / scale
                if compute_loss(U_r, V_r, W_r, target) < 1e-6:
                    return (U_r.tolist(), V_r.tolist(), W_r.tolist())
            if best_loss < 1e-6:
                return (best_U.tolist(), best_V.tolist(), best_W.tolist())
        return None

    # Try ranks from low to high
    for r in [9, 10, 11, 12, 13]:
        if time.time() > deadline - 30:
            break
        result = try_rank(r)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}

    return schoolbook()
# EVOLVE-BLOCK-END

