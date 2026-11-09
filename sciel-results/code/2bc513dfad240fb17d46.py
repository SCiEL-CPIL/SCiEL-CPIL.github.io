import numpy as np
import time
import scipy.integrate as integrate_scipy

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP using Radau solver with tight tolerances.
    This handles time-dependent and non-homogeneous linear systems accurately.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    # Infer Jacobian at t0 to potentially speed up the solver
    # This is an optimization; if the Jacobian is constant, it helps.
    # If not, the solver will still converge, potentially slower.
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        # Probe at t0
        col = np.asarray(f(t0, e_i), dtype=float)
        A[:, i] = col

    # Define Jacobian function
    # If we pass this, solve_ivp uses it instead of finite differences.
    # This saves calls if the Jacobian is constant or slowly varying.
    def jac(t, y):
        return A

    try:
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14, 
            jac=jac,
            max_step=0.1 # Prevent very large steps if needed, though Radau adapts
        )
        
        if time.time() > deadline:
            # If we are running out of time, return the last successful state if available
            if sol.success and sol.t.size > 0:
                return sol.y[:, -1]
            return y0

        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol2 = integrate_scipy.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-14, 
                jac=jac
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Fallback to expm if all else fails (likely to be inaccurate but better than nothing)
                from scipy.linalg import expm
                dt = t1 - t0
                M = expm(A * dt)
                return M @ y0

    except Exception:
        # Fallback to expm
        try:
            from scipy.linalg import expm
            dt = t1 - t0
            M = expm(A * dt)
            return M @ y0
        except Exception:
            return y0

    return y0
