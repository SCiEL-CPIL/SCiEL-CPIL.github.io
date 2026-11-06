"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def _rk4(f, t0, t1, y0, steps):
    """Fallback RK4 solver if the system is not linear."""
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
    Solve y' = f(t, y) using analytic matrix exponential for linear systems.
    Falls back to RK4 if linearity is not detected.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    # Try to detect linear time-invariant system: y' = A*y + b
    # Probe b
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    
    # Probe A columns
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        v_j = f(t0, e_j)
        A[:, j] = v_j - b
        
    # Verify linearity with a random test vector
    # Use a deterministic random vector to be reproducible
    np.random.seed(42)
    y_test = np.random.randn(n)
    v_test = f(t0, y_test)
    v_pred = A @ y_test + b
    
    # Check if linear
    # Tolerance should be tight relative to the magnitude
    norm_diff = np.linalg.norm(v_test - v_pred)
    norm_scale = max(1.0, np.linalg.norm(v_test), np.linalg.norm(v_pred))
    
    is_linear = norm_diff < 1e-10 * norm_scale
    
    if is_linear:
        dt = t1 - t0
        # Compute matrix exponential
        E = expm(A * dt)
        
        y_sol = E @ y0
        
        # If there is a constant term b, add the integral of exp(A*t) * b
        if np.linalg.norm(b) > 1e-14:
            # Integral is A^{-1} (E - I) b
            # Solve A * Z = E - I
            I_mat = np.eye(n)
            try:
                Z = np.linalg.solve(A, E - I_mat)
                y_sol = y_sol + Z @ b
            except np.linalg.LinAlgError:
                # If A is singular, use least squares or series
                # Series: sum_{k=0}^inf (A*dt)^k / (k+1)! * dt
                # This might be slow if eigenvalues are large, but dt is small (0.05)
                # Let's try lstsq first
                Z, _, _, _ = np.linalg.lstsq(A, E - I_mat, rcond=None)
                y_sol = y_sol + Z @ b
        return y_sol
    else:
        # Fallback to RK4
        # Use a reasonable number of steps for accuracy
        # 1000 steps for RK4 is usually very accurate for smooth problems
        y = _rk4(f, t0, t1, y0, 1000)
        if y is None:
             # Last resort: smaller steps
             y = _rk4(f, t0, t1, y0, 5000)
        return y

# EVOLVE-BLOCK-END
