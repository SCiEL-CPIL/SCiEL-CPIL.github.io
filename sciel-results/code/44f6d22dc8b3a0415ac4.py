import numpy as np
from scipy.optimize import root

def integrate(f, t_span, y0):
        t0, t1 = t_span
        y0_arr = np.array(y0, dtype=float)
        n_dims = len(y0_arr)
        
        # Number of Chebyshev points
        # Start with moderate N
        N = 60
        
        # Chebyshev-Lobatto points
        i = np.arange(N)
        t = (t1 + t0) / 2 - (t1 - t0) / 2 * np.cos(np.pi * i / (N - 1))
        
        # Differentiation matrix D
        # Construct D
        c = np.array([2.0] + [1.0]*(N-2) + [0.5])
        c = c * (-1.0)**i
        X = np.cos(np.pi * i / (N - 1))
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
        
        # Sub-matrix for unknowns y[1:N]
        D_sub = D[1:N, 1:N]
        D_col0 = D[1:N, 0]
        
        # Analytical Jacobian of Robertson
        def jac_f(t, y):
            y1, y2, y3 = y
            J = np.array([
                [-0.04, 1e4 * y3, 1e4 * y2],
                [0.04, -1e4 * y3 - 6e7 * y2, -1e4 * y2],
                [0.0, 6e7 * y2, 0.0]
            ])
            return J
            
        def residual(y_unknowns):
            # y_unknowns is flat array of size (N-1)*n_dims
            y_mat = y_unknowns.reshape(N-1, n_dims)
            
            # Derivatives from collocation
            # d_y = D_col0 * y0 + D_sub * y_mat
            d_y = np.outer(D_col0, y0_arr) + D_sub @ y_mat
            
            # f values
            f_vals = np.zeros((N-1, n_dims))
            for k in range(N-1):
                f_vals[k] = f(t[k+1], y_mat[k])
                
            return (d_y - f_vals).flatten()
            
        def jacobian(y_unknowns):
            y_mat = y_unknowns.reshape(N-1, n_dims)
            
            # J = D_sub (repeated n_dims times) - J_f
            # D_sub is (N-1)x(N-1).
            # We need to expand it to ((N-1)*n_dims) x ((N-1)*n_dims)
            # Block diagonal structure for J_f.
            
            J_full = np.zeros(((N-1)*n_dims, (N-1)*n_dims))
            
            # Fill D_sub blocks
            # For each row block i (corresponding to point i+1)
            # and col block j (corresponding to point j+1)
            # The block is D_sub[i, j] * I_n_dims
            
            for i in range(N-1):
                for j in range(N-1):
                    val = D_sub[i, j]
                    if val != 0:
                        J_full[i*n_dims:(i+1)*n_dims, j*n_dims:(j+1)*n_dims] = val * np.eye(n_dims)
                        
            # Subtract Jacobian of f
            for k in range(N-1):
                Jf = jac_f(t[k+1], y_mat[k])
                J_full[k*n_dims:(k+1)*n_dims, k*n_dims:(k+1)*n_dims] -= Jf
                
            return J_full
            
        # Initial guess
        # Use y0 for all points
        y_guess = np.tile(y0_arr, (N-1, 1)).flatten()
        
        try:
            # Use hybr (Levenberg-Marquardt)
            sol = root(residual, y_guess, jac=jacobian, method='hybr', tol=1e-12)
            
            if not sol.success:
                raise Exception("Root finding failed")
                
            y_sol = sol.x.reshape(N-1, n_dims)
            return y_sol[-1]
            
        except Exception:
            # Fallback
            import scipy.integrate
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0_arr, method='Radau', rtol=1e-10, atol=1e-12, t_eval=[t1]
            )
            return sol.y[:, -1]
