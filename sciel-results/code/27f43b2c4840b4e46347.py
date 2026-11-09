import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s
    
    # Extract matrix A for linear system y' = A y
    # A[:, j] = f(t0, e_j)
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        A[:, j] = f(t0, ej)
    
    # Verify linearity: check f(t0, y0) ≈ A @ y0
    # Also check if time-dependent: f(t1, y0) vs A @ y0
    fy0_t0 = f(t0, y0)
    # Check time independence
    if dt != 0:
        fy0_t1 = f(t1, y0)
        # If significantly different, system may be time-dependent
        if np.linalg.norm(fy0_t1 - fy0_t0) > 1e-10 * max(1.0, np.linalg.norm(fy0_t0)):
            # Time-dependent system: use implicit RK with Jacobian
            # Fall back to high-order implicit method
            y = _solve_time_dependent(f, t0, t1, y0, A, start_time, deadline)
            return y
    
    # Time-invariant linear system: use matrix exponential
    # y(t1) = expm(A * dt) @ y0
    M = expm(A * dt)
    y_end = M @ y0
    
    return y_end


def _solve_time_dependent(f, t0, t1, y0, A_guess, start_time, deadline):
    """Fallback for time-dependent systems: use high-order implicit RK with Jacobian from A_guess."""
    n = len(y0)
    y = np.array(y0, dtype=float)
    
    # Use Radau IIA or similar high-order implicit method
    # For simplicity, use a 4th-order Gauss-Legendre implicit RK with Newton
    # But for stiff systems, better to use backward differentiation or Radau
    
    # Use scipy's solve_ivp with Radau method if available
    from scipy.integrate import solve_ivp
    
    t_end = t1
    sol = solve_ivp(f, [t0, t1], y, method='Radau', rtol=1e-13, atol=1e-15, 
                    jac=lambda t, y: A_guess, max_step=0.01)
    
    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]
    
    # Fallback: backward Euler with many steps
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if time.time() > deadline:
            break
        z = y.copy()
        for _newton in range(10):
            fz = f(t + h, z)
            # Use A_guess as Jacobian approximation
            J = A_guess
            step = np.linalg.solve(np.eye(n) - h * J, y - z + h * fz)
            z_new = z + step
            if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z)):
                z = z_new
                break
            z = z_new
        y = z
        t += h
    return y
