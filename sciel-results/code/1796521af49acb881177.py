"""Heat2D 10x10 ODE solver using matrix exponential approach."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy: The heat equation discretized on a 10x10 grid gives a linear,
    autonomous, stiff ODE: y' = A*y. We estimate A using a small number of
    function evaluations, then compute the exact solution y(t_end) = expm(A*(t_end-t0)) @ y0.
    
    Since the system is linear and autonomous, we can:
    1. Estimate A by evaluating f at y0 and perturbing in coordinate directions
    2. Use scipy.linalg.expm for the matrix exponential
    3. Return expm(A * dt) @ y0 as the final state
    
    This gives essentially zero integration error (limited only by numerical
    precision of expm) with O(n^2) function evaluations where n = len(y0).
    """
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Estimate the Jacobian A = df/dy at t0 (for autonomous linear system, this is constant)
    # Use forward differences with a small perturbation
    
    # First, get f(t0, y0)
    f0 = np.asarray(f(t0, y0), dtype=float)
    
    # Estimate A using forward differences
    # For a linear system y' = A*y, we have f(t, y) = A*y (possibly + g(t) but for heat eq it's homogeneous)
    # So A = f(t, y) / y component-wise... but that's not quite right for a matrix.
    # Better: perturb each coordinate and compute (f(y+e_j*eps) - f(y)) / (eps)
    
    eps = 1e-8
    A = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        if time.time() > deadline:
            break
        y_pert = y0.copy()
        y_pert[j] += eps
        f_pert = np.asarray(f(t0, y_pert), dtype=float)
        A[:, j] = (f_pert - f0) / eps
    
    # For a linear autonomous system, the solution is y(t) = expm(A*(t-t0)) @ y0
    # But we need to verify this is indeed linear and autonomous.
    # Check: is f(t0, 2*y0) ≈ 2*f(t0, y0)? (linearity check)
    # And is f(t0+dt/2, y0) ≈ f(t0, y0)? (autonomy check)
    
    # Let's do a quick linearity check with one extra evaluation
    f_double = np.asarray(f(t0, 2 * y0), dtype=float)
    linearity_error = np.linalg.norm(f_double - 2 * f0) / (2 * np.linalg.norm(f0) + 1e-30)
    
    # Autonomy check
    f_mid = np.asarray(f(t0 + dt / 2, y0), dtype=float)
    autonomy_error = np.linalg.norm(f_mid - f0) / (np.linalg.norm(f0) + 1e-30)
    
    if linearity_error < 1e-6 and autonomy_error < 1e-6:
        # System is linear and autonomous: use exact matrix exponential
        # y(t1) = expm(A * dt) @ y0
        if time.time() > deadline:
            # Fallback to a simple RK4 if we're running out of time
            h = dt / 1000
            t = t0
            y = y0.copy()
            for _ in range(1000):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y
        else:
            M = A * dt
            expM = expm(M)
            y_final = expM @ y0
            return y_final
    else:
        # Not linear/autonomous as expected; fall back to high-order RK4
        # Use a larger number of steps for accuracy
        steps = 5000
        h = dt / steps
        t = t0
        y = y0.copy()
        for i in range(steps):
            if time.time() > deadline:
                break
            k1 = np.asarray(f(t, y), dtype=float)
            k2 = np.asarray(f(t + h / 2, y + h / 2 * k1), dtype=float)
            k3 = np.asarray(f(t + h / 2, y + h / 2 * k2), dtype=float)
            k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
# EVOLVE-BLOCK-END
