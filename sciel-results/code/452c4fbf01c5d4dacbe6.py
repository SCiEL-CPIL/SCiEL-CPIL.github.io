"""
Optimized solver for the 10x10 2D Heat Equation.
Uses the linearity of the problem to compute the exact solution via matrix exponential,
minimizing the number of function evaluations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 250.0  # Safety margin before 300s/330s limit


def _get_matrix_A(f, t0, n):
    """
    Reconstruct the linear operator A by probing f with basis vectors.
    Assumes y' = A y.
    """
    # Check for affine term f(t, 0)
    y_zero = np.zeros(n)
    f_zero = f(t0, y_zero)
    
    # If f(0) is not zero, the system is affine: y' = Ay + b
    # We can handle this by augmenting the state or just noting b.
    # For the heat equation, b is typically 0.
    # If b is not 0, we can still solve y' = Ay + b(t) if b is constant or linear.
    # Let's assume b is constant for now. If it's not, we might need to integrate b.
    # Actually, for heat equation with Dirichlet BC, the system is homogeneous.
    
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
        
    return A, f_zero


def _solve_linear_exponential(f, t_span, y0):
    """
    Solve y' = A y + b using matrix exponential.
    y(t) = expm(A*(t-t0)) y0 + integral from t0 to t of expm(A*(t-s)) b ds
    If b is constant: y(t) = expm(A*(t-t0)) y0 + (expm(A*(t-t0)) - I) A^{-1} b
    """
    t0, t1 = t_span
    dt = t1 - t0
    n = len(y0)
    
    A, b = _get_matrix_A(f, t0, n)
    
    # Compute matrix exponential
    M = expm(A * dt)
    
    y_final = M @ y0
    
    # Add contribution from b if it's not zero
    if np.linalg.norm(b) > 1e-10:
        # Check if b is constant by evaluating at another time
        b_mid = f(t0 + dt/2, np.zeros(n))
        if np.linalg.norm(b_mid - b) < 1e-8 * (1 + np.linalg.norm(b)):
            # b is constant
            try:
                A_inv = np.linalg.inv(A)
                y_final += (M - np.eye(n)) @ A_inv @ b
            except np.linalg.LinAlgError:
                # A is singular, use numerical integration
                # y(t) = M y0 + integral_0^dt M(-s) b ds
                # Use Simpson's rule
                n_steps = 100
                s = np.linspace(0, dt, n_steps+1)
                integral = np.zeros(n)
                for i in range(n_steps+1):
                    if i == 0 or i == n_steps:
                        w = 1
                    elif i % 2 == 0:
                        w = 2
                    else:
                        w = 4
                    integral += w * expm(-A * s[i]) @ b
                integral *= dt / (3 * n_steps)
                y_final += integral
    
    return y_final


def _rk4(f, t0, t1, y0, steps):
    """Standard RK4 solver."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    """
    Main entry point. Tries the linear exponential method first,
    falls back to RK4 if it fails or takes too long.
    """
    t0, t1 = t_span
    
    # Try the exact linear solver
    try:
        if time.time() - _START_TIME < _TIMEOUT:
            y = _solve_linear_exponential(f, t_span, y0)
            if y is not None and np.all(np.isfinite(y)):
                return y
    except Exception:
        pass
    
    # Fall back to RK4
    y = _rk4(f, t0, t1, y0, 2000)
    if y is not None:
        return y
    
    # Final fallback: smaller steps
    y = _rk4(f, t0, t1, y0, 10000)
    return y
# EVOLVE-BLOCK-END
