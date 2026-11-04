import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def compute_error(U, V, W):
        r = U.shape[1]
        res = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            # res[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        return np.sum((res - 1.0) ** 2)

    def verify_and_snap(U, V, W, tol=1e-9):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        r = U_snap.shape[1]
        res = np.zeros((n, m, p))
        for c in range(r):
            Uc = U_snap[:, c].reshape(n, m)
            Vc = V_snap[:, c].reshape(m, p)
            Wc = W_snap[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        if np.allclose(res, 1.0, atol=tol):
            return U_snap, V_snap, W_snap
        return None

    def solve_U(U, V, W):
        r = U.shape[1]
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    U_new[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        return U_new

    def solve_V(U, V, W):
        r = V.shape[1]
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    V_new[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        return V_new

    def solve_W(U, V, W):
        r = W.shape[1]
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    W_new[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        return W_new

    fallback = schoolbook()
    best_result = None
    best_rank = n * m * p

    # Try ranks from low to high
    for r in range(20, best_rank):
        if time.time() > deadline:
            break

        time_left = deadline - time.time()
        if time_left < 3.0:
            break

        # Number of restarts: spend ~1.5s per restart on average
        max_restarts = max(1, int(time_left / 1.5))

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Initialize with small random values
            scale = 1.0 / np.sqrt(r)
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale

            # ALS iterations
            converged = False
            prev_error = np.inf
            for it in range(200):
                if time.time() > deadline:
                    break

                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)

                error = compute_error(U, V, W)

                if error < 1e-10:
                    converged = True
                    break

                # Early stop if not improving
                if error > prev_error * 1.1 and it > 10:
                    break
                prev_error = error

            if time.time() > deadline:
                break

            if converged:
                result = verify_and_snap(U, V, W)
                if result:
                    U_sol, V_sol, W_sol = result
                    return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}

    return fallback
