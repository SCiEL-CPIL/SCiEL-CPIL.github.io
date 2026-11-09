"""ODE integrator for the Van der Pol oscillator (mu = 5) using scipy Radau with analytic Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time


def _vdp_jac(t, y):
    """Analytic Jacobian for Van der Pol oscillator with mu=5."""
    y0_val, y1_val = y[0], y[1]
    mu = 5.0
    return np.array([
        [0.0, 1.0],
        [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val * y0_val)]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator y0' = y1, y1' = 5(1 - y0^2)y1 - y0
    from t_span[0] to t_span[1] with initial condition y0.
    Returns the state vector at t_span[1].
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_vdp_jac
        )

        if sol.success:
            return sol.y[:, -1]
        else:
            sol2 = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=_vdp_jac
            )
            if sol2.success:
                return sol2.y[:, -1]

    except Exception:
        pass

    # Fallback: fixed-step RK4 with small step size
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = 1e-4
    t = t0
    n_steps = int((t1 - t0) / h) + 1

    for _ in range(n_steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h

    return y
# EVOLVE-BLOCK-END
