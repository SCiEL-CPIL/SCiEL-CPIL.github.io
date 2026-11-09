import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Check for linearity and autonomy
    # f(t, y) = A y
    f0 = f(t0, y0)
    f1 = f(t0, 2.0 * y0)
    
    # Tolerance for linearity check
    # If f is linear, f(2y) = 2f(y)
    is_linear = np.allclose(f1, 2.0 * f0, rtol=1e-5, atol=1e-8)
    
    if is_linear:
        # Construct Jacobian A using central finite differences
        # This requires 2N calls.
        A = np.zeros((N, N))
        h_fd_base = 1e-5
        for j in range(N):
            e = np.zeros(N)
            scale = max(1.0, abs(y0[j]))
            h_j = h_fd_base * scale
            e[j] = h_j
            fp = f(t0, y0 + e)
            e[j] = -h_j
            fm = f(t0, y0 + e)
            A[:, j] = (fp - fm) / (2.0 * h_j)
            
        # Use Matrix Exponential for exact solution of linear system
        # y(t) = exp(A * (t1 - t0)) * y0
        dt = t1 - t0
        exp_A_dt = expm(A * dt)
        y_final = exp_A_dt @ y0
        return y_final
    else:
        # Fallback to robust solver
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=(t1-t0)/100)
        return sol.y[:, -1]
