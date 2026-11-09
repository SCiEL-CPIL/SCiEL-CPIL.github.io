import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit

    n, m, p = 3, 3, 5
    size_U = n * m  # 9
    size_V = m * p  # 15
    size_W = p * n  # 15

    # Trivial rank-45 decomposition
    R0 = n * m * p  # 45
    U0 = np.zeros((size_U, R0))
    V0 = np.zeros((size_V, R0))
    W0 = np.zeros((size_W, R0))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1

    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = R0

    # Precompute index triples for efficient residual computation
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def compute_residual(Ux, Vx, Wx):
        """Compute sum of squared residuals: for each (i,j,k), need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1."""
        residual = 0.0
        for (i, j, k) in triples:
            u_row = Ux[i*m+j, :]
            v_row = Vx[j*p+k, :]
            w_row = Wx[k*n+i, :]
            val = np.dot(u_row, v_row * w_row)
            residual += (1.0 - val) ** 2
        return residual

    def verify(Ux, Vx, Wx, tol=1e-9):
        """Check exact decomposition with half-integer entries."""
        for (i, j, k) in triples:
            u_row = Ux[i*m+j, :]
            v_row = Vx[j*p+k, :]
            w_row = Wx[k*n+i, :]
            val = np.dot(u_row, v_row * w_row)
            if abs(val - 1.0) > tol:
                return False
        return True

    # Strategy: Try to find lower rank by starting from the trivial solution
    # and using structured perturbations + optimization
    # Key insight: the trivial decomposition has many symmetries; we can
    # try to merge columns by finding linear dependencies

    for R in range(R0 - 1, max(R0 - 6, 30), -1):
        if time.time() - start_time > timeout - 15:
            break

        found = False
        for attempt in range(8):
            if time.time() - start_time > timeout - 10:
                break

            np.random.seed(1000 + R * 10 + attempt)

            # Initialize: take R columns from the trivial solution with random mixing
            # This preserves the structure better than pure random
            if R == R0 - 1:
                # Remove one column and add small perturbation to others
                U = U0[:, :R].copy()
                V = V0[:, :R].copy()
                W = W0[:, :R].copy()
                # Small perturbation to escape exact degeneracy
                U += 0.001 * np.random.randn(size_U, R)
                V += 0.001 * np.random.randn(size_V, R)
                W += 0.001 * np.random.randn(size_W, R)
            else:
                # Mix columns randomly to create a different starting point
                perm = np.random.permutation(R0)
                U = U0[:, perm[:R]].copy()
                V = V0[:, perm[:R]].copy()
                W = W0[:, perm[:R]].copy()
                U += 0.01 * np.random.randn(size_U, R)
                V += 0.01 * np.random.randn(size_V, R)
                W += 0.01 * np.random.randn(size_W, R)

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def objective(x, R=R):
                if time.time() - start_time > timeout - 5:
                    return 1e10
                Ux = x[:size_U*R].reshape(size_U, R)
                Vx = x[size_U*R:2*size_U*R].reshape(size_V, R)
                Wx = x[2*size_U*R:].reshape(size_W, R)
                return compute_residual(Ux, Vx, Wx)

            try:
                # Use L-BFGS-B with tight settings
                result = minimize(
                    objective, x0, method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-12}
                )

                if result.fun < 1e-8:
                    Ux = result.x[:size_U*R].reshape(size_U, R)
                    Vx = result.x[size_U*R:2*size_U*R].reshape(size_V, R)
                    Wx = result.x[2*size_U*R:].reshape(size_W, R)

                    # Try rounding to half-integers
                    Ux_r = np.round(Ux * 2) / 2
                    Vx_r = np.round(Vx * 2) / 2
                    Wx_r = np.round(Wx * 2) / 2

                    if verify(Ux_r, Vx_r, Wx_r_r := Wx_r, tol=1e-9):
                        best_result = {"U": Ux_r.tolist(), "V": Vx_r.tolist(), "W": Wx_r.tolist()}
                        best_rank = R
                        found = True
                        break

                    # Also try integer rounding
                    Ux_i = np.round(Ux)
                    Vx_i = np.round(Vx)
                    Wx_i = np.round(Wx)
                    if verify(Ux_i, Vx_i, Wx_i, tol=1e-9):
                        best_result = {"U": Ux_i.tolist(), "V": Vx_i.tolist(), "W": Wx_i.tolist()}
                        best_rank = R
                        found = True
                        break

            except Exception:
                pass

        if found:
            break

    return best_result
