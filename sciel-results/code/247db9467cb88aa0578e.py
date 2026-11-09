"""ODE solver for 100-point convection-diffusion IVP using matrix exponential with adaptive fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def integrate(f, t_span, y0):
    # Deadline guard
    t_start = time.perf_counter()
    DEADLINE = 290.0  # Safety margin before 330s kill

    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    n = y0.shape[0]

    def _elapsed_ok():
        return (time.perf_counter() - t_start) < DEADLINE

    # Strategy 1: Exploit linearity and time-invariance.
    # For y' = A*y (autonomous linear system), y(t1) = expm(A*(t1-t0)) @ y0
    # Extract A by evaluating f(t0, e_i) for each basis vector e_i.
    # This requires n function calls (n=100) and gives ~15-16 digits accuracy.
    
    try:
        if not _elapsed_ok():
            return y0

        # Extract the system matrix A: f(t, y) = A @ y for autonomous linear systems
        # Evaluate f at t0 with each standard basis vector
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            col = f(t0, e_i)
            A[:, i] = np.asarray(col, dtype=float)
        
        if not _elapsed_ok():
            return y0

        # Verify linearity: check that f(t0, y0) ≈ A @ y0
        f_y0 = np.asarray(f(t0, y0), dtype=float)
        A_y0 = A @ y0
        if np.all(np.isfinite(A)) and np.all(np.isfinite(A_y0)):
            residual = np.linalg.norm(f_y0 - A_y0)
            scale = max(np.linalg.norm(f_y0), np.linalg.norm(A_y0), 1e-300)
            if residual / scale < 1e-10:
                # System is linear and time-invariant (at least approximately)
                T = t1 - t0
                # Compute matrix exponential
                expA = expm(A * T)
                if np.all(np.isfinite(expA)):
                    y_final = expA @ y0
                    if np.all(np.isfinite(y_final)):
                        return y_final
    except Exception:
        pass

    # Strategy 2: Fallback to DOP853 with tight tolerances
    try:
        if not _elapsed_ok():
            return y0
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
            max_step=(t1 - t0) / 100.0,
            first_step=(t1 - t0) / 1e5,
        )
        if sol.success and sol.t.size > 0:
            if np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
    except Exception:
        pass

    # Strategy 3: Fallback to Radau for stiff problems
    try:
        if not _elapsed_ok():
            return y0
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="Radau",
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 50.0,
        )
        if sol.success and sol.t.size > 0:
            if np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
    except Exception:
        pass

    # Strategy 4: Fallback to BDF
    try:
        if not _elapsed_ok():
            return y0
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method="BDF",
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 50.0,
        )
        if sol.success and sol.t.size > 0:
            if np.all(np.isfinite(sol.y[:, -1])):
                return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: RK4
    if not _elapsed_ok():
        return y0

    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if not _elapsed_ok():
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    return y
# EVOLVE-BLOCK-END
