import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Probe for constant linear system: y' = A*y + b
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)
        
    # Verification
    y_test = np.ones(n)
    t_test = t0 + 0.5 * (t1 - t0)
    f_test = np.array(f(t_test, y_test), dtype=float)
    f_pred = A @ y_test + b
    
    if np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-12):
        dt = t1 - t0
        
        # Augmented State Method:
        # Let z = [y; t]. Then z' = [A b; 0 0] z.
        # z(t) = expm([A b; 0 0] * (t-t0)) * z(t0)
        # This avoids explicit inversion of A and subtraction of matrices.
        
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = A
        A_aug[:n, n] = b
        
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        # z0[n] = 0
        
        M_aug = expm(A_aug * dt)
        z_end = M_aug @ z0
        
        return z_end[:n]

    else:
        # Fallback for non-linear or time-varying systems
        def jac(t, y):
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-8 * max(1.0, abs(y[j]))
                J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
            return J

        sol = solve_ivp(f, t_span, y0, method='Radau', jac=jac, rtol=1e-12, atol=1e-14)
        if not sol.success:
             sol = solve_ivp(f, t_span, y0, method='BDF', jac=jac, rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
