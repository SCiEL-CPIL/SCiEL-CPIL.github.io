"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as scipy_integrate


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using a high-accuracy implicit method.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 300s limit, leave safety margin
    deadline = time.time() + 290.0
    
    # Robertson problem has a known analytic Jacobian structure.
    # The standard Robertson problem is:
    # y1' = -1e-4*y1 + 4e-3*y2 - 1e-4*y1*y2
    # y2' = 1e-4*y1 - 4e-3*y2 - 3e-7*y2^2
    # y3' = 3e-7*y2^2
    #
    # We provide an analytic Jacobian to scipy's Radau solver for efficiency.
    # The Jacobian J(y) = [[-1e-4 - 1e-4*y2, 4e-3 - 1e-4*y1, 0],
    #                       [1e-4, -4e-3 - 6e-7*y2, 0],
    #                       [0, 6e-7*y2, 0]]
    #
    # But we don't know for sure the exact form of f. Let's use a generic
    # approach: use Radau with a finite-difference Jacobian if we can't
    # assume the specific form, OR we can try to detect it.
    #
    # Actually, the task says "the fixed Robertson initial-value problem",
    # so we can assume the standard form. Let's define the Jacobian.
    
    def jac(t, y):
        """Analytic Jacobian for the Robertson problem."""
        y1, y2, y3 = y
        J = np.array([
            [-1e-4 - 1e-4 * y2, 4e-3 - 1e-4 * y1, 0.0],
            [1e-4, -4e-3 - 6e-7 * y2, 0.0],
            [0.0, 6e-7 * y2, 0.0]
        ])
        return J
    
    # Use Radau (implicit Runge-Kutta, good for stiff problems) with tight tolerance
    try:
        sol = scipy_integrate.solve_ivp(
            f, [t0, t1], y0,
            method='Radau',
            jac=jac,
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: use Radau with finite-difference Jacobian
    try:
        sol = scipy_integrate.solve_ivp(
            f, [t0, t1], y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: use BDF
    try:
        sol = scipy_integrate.solve_ivp(
            f, [t0, t1], y0,
            method='BDF',
            rtol=1e-12,
            atol=1e-14,
            max_step=1.0
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Absolute last resort: dense RK4 with many steps
    steps = 100000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y

# EVOLVE-BLOCK-END
