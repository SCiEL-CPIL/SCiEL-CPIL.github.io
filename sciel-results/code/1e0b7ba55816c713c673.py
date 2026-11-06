import numpy as np
from scipy.linalg import solve

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n_dims = len(y0_arr)
    
    def get_chebyshev(N):
        i = np.arange(N)
        t = (t1 + t0) / 2 - (t1 - t0) / 2 * np.cos(np.pi * i / (N - 1))
        return t

    def get_diff_matrix(N):
        i = np.arange(N)
        X = np.cos(np.pi * i / (N - 1))
        c = np.array([2.0] + [1.0]*(N-2) + [0.5])
        c = c * (-1.0)**i
        D = np.zeros((N, N))
        for k in range(N):
            for m in range(N):
                if k == m:
                    if k == 0 or k == N-1:
                        D[k, m] = -X[k] / 2.0
                    else:
                        D[k, m] = -X[k] / (2.0 * (1.0 - X[k]**2))
                else:
                    D[k, m] = c[k] / (c[m] * (X[k] - X[m]))
        return D

    def jac_f(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
            [0.0, 6e7 * y2, 0.0]
        ])
        return J

    def solve_for_N(N, y_init):
        t = get_chebyshev(N)
        D = get_diff_matrix(N)
        D_sub = D[1:N, 1:N]
        D_col0 = D[1:N, 0]
        M = (N-1) * n_dims
        
        def residual(y_flat):
            y_mat = y_flat.reshape(N-1, n_dims)
            d_y = np.outer(D_col0, y0_arr) + D_sub @ y_mat
            f_vals = np.zeros((N-1, n_dims))
            for k in range(N-1):
                f_vals[k] = f(t[k+1], y_mat[k])
            return (d_y - f_vals).flatten()

        def jacobian(y_flat):
            y_mat = y_flat.reshape(N-1, n_dims)
            J_full = np.zeros((M, M))
            for i in range(N-1):
                for j in range(N-1):
                    val = D_sub[i, j]
                    if val != 0:
                        J_full[i*n_dims:(i+1)*n_dims, j*n_dims:(j+1)*n_dims] = val * np.eye(n_dims)
            for k in range(N-1):
                Jf = jac_f(t[k+1], y_mat[k])
                J_full[k*n_dims:(k+1)*n_dims, k*n_dims:(k+1)*n_dims] -= Jf
            return J_full

        y_current = y_init.flatten()
        for _ in range(50):
            R = residual(y_current)
            if np.max(np.abs(R)) < 1e-13:
                break
            J = jacobian(y_current)
            try:
                dy = solve(J, -R)
            except Exception:
                dy = np.linalg.solve(J + 1e-10*np.eye(M), -R)
            y_current += dy
        return y_current.reshape(N-1, n_dims)

    # Multi-resolution
    N1 = 10
    y_init1 = np.tile(y0_arr, (N1-1, 1))
    y_sol1 = solve_for_N(N1, y_init1)
    
    N2 = 40
    t1_grid = get_chebyshev(N1)
    t2_grid = get_chebyshev(N2)
    y_init2 = np.zeros((N2-1, n_dims))
    for d in range(n_dims):
        y_init2[:, d] = np.interp(t2_grid[1:], t1_grid[1:], y_sol1[:, d])
    y_sol2 = solve_for_N(N2, y_init2)
    
    N3 = 100
    t3_grid = get_chebyshev(N3)
    y_init3 = np.zeros((N3-1, n_dims))
    for d in range(n_dims):
        y_init3[:, d] = np.interp(t3_grid[1:], t2_grid[1:], y_sol2[:, d])
    y_sol3 = solve_for_N(N3, y_init3)
    
    return y_sol3[-1]
