import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 120.0

    n, m, p = DIMENSIONS

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

    def verify(U, V, W, r, tol=1e-9):
        """Verify that the decomposition is exact: T[i,j,k] = sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] = 1 for all i,j,k"""
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijs,jks,kis->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, 1.0, atol=tol)

    def round_to_half_int(U, V, W, r):
        """Round to half-integers and verify"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, r, tol=1e-8):
            return U_r, V_r, W_r
        return None

    def try_rank(r, max_restarts=20, seed_base=0):
        """Try to find a rank-r decomposition using L-BFGS-B optimization"""
        U_size = n * m * r
        V_size = m * p * r
        W_size = p * n * r
        total_size = U_size + V_size + W_size

        for restart in range(max_restarts):
            if time.time() - start_time > timeout - 15:
                return None

            np.random.seed(seed_base + restart)

            # Different initialization strategies
            strategy = restart % 4
            if strategy == 0:
                scale = 0.5
            elif strategy == 1:
                scale = 1.0
            elif strategy == 2:
                scale = 0.3
            else:
                scale = 0.7

            U_init = np.random.randn(n * m, r) * scale
            V_init = np.random.randn(m * p, r) * scale
            W_init = np.random.randn(p * n, r) * scale

            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            def loss_and_grad(params):
                U_flat = params[:U_size]
                V_flat = params[U_size:U_size + V_size]
                W_flat = params[U_size + V_size:]

                U = U_flat.reshape(n * m, r)
                V = V_flat.reshape(m * p, r)
                W = W_flat.reshape(p * n, r)

                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)

                T = np.einsum('ijs,jks,kis->ijk', U_3d, V_3d, W_3d)
                loss = np.sum((T - 1.0)**2)

                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = T[i, j, k] - 1.0
                            if abs(diff) < 1e-12:
                                continue
                            grad_U[i, j, :] += 2.0 * diff * (V_3d[j, k, :] * W_3d[k, i, :])
                            grad_V[j, k, :] += 2.0 * diff * (U_3d[i, j, :] * W_3d[k, i, :])
                            grad_W[k, i, :] += 2.0 * diff * (U_3d[i, j, :] * V_3d[j, k, :])

                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return loss, grad

            try:
                import scipy.optimize as opt
                res = opt.minimize(
                    lambda x: loss_and_grad(x)[0],
                    params_init,
                    jac=lambda x: loss_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )

                if res.fun < 1e-8:
                    U_flat = res.x[:U_size]
                    V_flat = res.x[U_size:U_size + V_size]
                    W_flat = res.x[U_size + V_size:]

                    U = U_flat.reshape(n * m, r)
                    V = V_flat.reshape(m * p, r)
                    W = W_flat.reshape(p * n, r)

                    result = round_to_half_int(U, V, W, r)
                    if result is not None:
                        U_r, V_r, W_r = result
                        return {
                            "U": [[float(x) for x in row] for row in U_r],
                            "V": [[float(x) for x in row] for row in V_r],
                            "W": [[float(x) for x in row] for row in W_r]
                        }
            except Exception:
                pass

        return None

    # Try ranks from 71 down to 40
    for r in range(71, 39, -1):
        if time.time() - start_time > timeout - 15:
            break

        result = try_rank(r, max_restarts=10, seed_base=r * 100)
        if result is not None:
            return result

    return schoolbook()
