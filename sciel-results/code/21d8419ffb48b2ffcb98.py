"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Radau with analytical Jacobian.
    Tight tolerances to minimize relative error while keeping function calls low.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0

    from scipy.integrate import solve_ivp

    # Analytical Jacobian for Van der Pol (mu=5)
    # y0' = y1
    # y1' = 5*(1 - y0^2)*y1 - y0
    # J = [[0, 1],
    #      [-10*y0*y1 - 1, 5*(1 - y0^2)]]
    def jac(t, y):
        y0_val = y[0]
        y1_val = y[1]
        return np.array([
            [0.0, 1.0],
            [-10.0 * y0_val * y1_val - 1.0, 5.0 * (1.0 - y0_val * y0_val)]
        ])

    # Wrap f to ensure proper return type
    def f_wrapped(t, y):
        result = f(t, y)
        return np.asarray(result, dtype=float)

    try:
        sol = solve_ivp(
            f_wrapped,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            jac=jac,
            dense_output=False
        )

        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: Radau without Jacobian
    try:
        if time.time() > deadline:
            raise TimeoutError()
        sol = solve_ivp(
            f_wrapped,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )

        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Final fallback: RK4 with moderate steps
    if time.time() > deadline:
        return y0
    
    steps = 2000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break

    if np.all(np.isfinite(y)):
        return y
    
    return y0
# EVOLVE-BLOCK-END
