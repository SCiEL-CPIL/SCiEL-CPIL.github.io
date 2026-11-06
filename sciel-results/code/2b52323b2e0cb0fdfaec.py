"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solves y' = f(t, y) for a 15-dimensional stiff linear system.
    Exploits linearity and time-invariance: estimate A, then use expm.
    Falls back to a stiff ODE solver if linearity/time-invariance is not detected.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    # Deadline guard: 330s wall-clock limit, use 320s safety margin
    start_time = time.time()
    deadline = start_time + 320.0

    # Step 1: Estimate the matrix A assuming f(t, y) = A @ y (autonomous linear).
    # Evaluate f at each standard basis vector at t0.
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)

        if time.time() > deadline:
            return y0

    # Step 2: Verify linearity and time-invariance.
    # Check: f(t0, y0) should equal A @ y0 (linearity at t0)
    # Check: f(t1, y0) should equal A @ y0 (time-invariance)
    f_t0_y0 = f(t0, y0)
    f_t1_y0 = f(t1, y0)

    norm_y0 = np.linalg.norm(y0)
    tol = 1e-10 * (1.0 + norm_y0)

    is_linear_t0 = np.linalg.norm(f_t0_y0 - A @ y0) < tol
    is_time_inv = np.linalg.norm(f_t1_y0 - A @ y0) < tol

    if is_linear_t0 and is_time_inv:
        # Confirmed: autonomous linear system. Use matrix exponential.
        dt = t1 - t0
        from scipy.linalg import expm

        try:
            expA_dt = expm(A * dt)
            y_final = expA_dt @ y0
            return y_final
        except Exception:
            pass

        # Fallback: eigenvalue-based matrix exponential
        try:
            eigvals, eigvecs = np.linalg.eig(A * dt)
            eig_inv = np.linalg.inv(eigvecs)
            y_final = eigvecs @ (np.exp(eigvals) * (eig_inv @ y0))
            return y_final
        except Exception:
            pass

    # Fallback: not confirmed as autonomous linear. Use a stiff ODE solver.
    # We've used n+2 calls already; plenty of budget remains.
    from scipy.integrate import solve_ivp

    try:
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-13, atol=1e-15)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    try:
        sol = solve_ivp(f, [t0, t1], y0, method='BDF', rtol=1e-13, atol=1e-15)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort
    return y0

# EVOLVE-BLOCK-END
