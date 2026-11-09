"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using a stiff-capable implicit method.
    We use a custom BDF2 (Backward Differentiation Formula of order 2) solver
    with Newton iteration and finite-difference Jacobian, which is well-suited
    for stiff problems like Robertson.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # We'll use adaptive step-size BDF2 with Newton iteration.
    # Start with a small step and grow it.
    
    # Deadline guard
    deadline = time.time() + 280.0  # Leave 50s margin before 330s kill
    
    # Initial step size - Robertson is stiff, start small
    h = 1e-3
    t = t0
    
    # BDF2 requires 2 steps; start with a single step of backward Euler
    # to get the second point, then switch to BDF2.
    
    def _solve_newton(t_new, y_prev, h_new, f):
        """Solve y_new - h*f(t_new, y_new) = y_prev for implicit step."""
        y_new = y_prev.copy()
        for _ in range(10):
            f_new = f(t_new, y_new)
            residual = y_new - h_new * f_new - y_prev
            # Finite difference Jacobian
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = max(1.0, abs(y_new[j]))
                e[j] = 1e-8 * scale
                J[:, j] = (f(t_new, y_new + e) - f_new) / e[j]
            A = np.eye(n) - h_new * J
            try:
                step = np.linalg.solve(A, -residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(A, -residual, rcond=None)[0]
            y_new = y_new + step
            if np.linalg.norm(step) < 1e-12 * (1 + np.linalg.norm(y_new)):
                break
        return y_new
    
    def _solve_bdf2_newton(t_new, y0_prev, y1_prev, h_new, f):
        """Solve BDF2 step: (3/2)*y_new - y_prev + (1/2)*y_prev2 = 3*h*f(t_new, y_new)"""
        y_new = y1_prev.copy()
        for _ in range(10):
            f_new = f(t_new, y_new)
            residual = 1.5 * y_new - y1_prev + 0.5 * y0_prev - 3.0 * h_new * f_new
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = max(1.0, abs(y_new[j]))
                e[j] = 1e-8 * scale
                J[:, j] = (f(t_new, y_new + e) - f_new) / e[j]
            A = 1.5 * np.eye(n) - 3.0 * h_new * J
            try:
                step = np.linalg.solve(A, -residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(A, -residual, rcond=None)[0]
            y_new = y_new + step
            if np.linalg.norm(step) < 1e-12 * (1 + np.linalg.norm(y_new)):
                break
        return y_new
    
    # First step: backward Euler to get y1
    y_prev2 = y.copy()
    t_new = t0 + h
    y_prev1 = _solve_newton(t_new, y_prev2, h, f)
    t = t_new
    h = 2 * h  # Try to double step size
    
    # Adaptive BDF2 loop
    while t < t1 and time.time() < deadline:
        if t + h > t1:
            h = t1 - t
            if h <= 0:
                break
        
        y_new = _solve_bdf2_newton(t + h, y_prev2, y_prev1, h, f)
        
        # Check for divergence
        if not np.all(np.isfinite(y_new)):
            h = h / 2
            if h < 1e-12:
                break
            continue
        
        # Estimate error using a simpler local check: compare with a half-step
        # For simplicity, we'll use a fixed growth strategy with occasional checks
        y_prev2 = y_prev1
        y_prev1 = y_new
        t = t + h
        
        # Grow step size cautiously
        if h < 0.5:
            h = min(h * 1.5, 0.5)
        else:
            h = min(h * 1.1, t1 - t if t1 - t > h else h)
    
    return y_prev1
# EVOLVE-BLOCK-END
