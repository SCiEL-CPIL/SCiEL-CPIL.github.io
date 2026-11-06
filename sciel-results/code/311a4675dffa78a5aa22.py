"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using an implicit Radau solver with analytic Jacobian.
    
    Parameters:
    -----------
    f : callable
        The ODE function f(t, y) -> y'
    t_span : tuple
        (t0, t1) integration interval
    y0 : array-like
        Initial condition
    
    Returns:
    --------
    y_end : ndarray
        Final state at t1
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 300s for safety
    start_time = time.time()
    time_limit = 300.0  # seconds
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J
    
    # Strategy: use Radau with progressively tighter tolerances
    # Score = -log10(rel_err) - 0.5*log10(nfev)
    # We want high accuracy (low rel_err) and low nfev
    
    best_result = None
    best_score = -np.inf
    
    # Tolerances to try (from moderate to very tight)
    # For Robertson, Radau is efficient with analytic Jacobian
    tolerance_configs = [
        (1e-10, 1e-14),   # tight
        (1e-12, 1e-16),   # very tight
        (1e-14, 1e-18),   # extremely tight
    ]
    
    for rtol, atol in tolerance_configs:
        if time.time() - start_time > time_limit - 10.0:
            break
        
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                jac=jac,
                rtol=rtol,
                atol=atol,
                max_step=1.0,
            )
            
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                break
            
            if sol.success:
                y_end = sol.y[:, -1]
                nfev = sol.nfev
                
                # Estimate relative error: for Robertson, the solution components
                # are roughly O(1), O(1e-5), O(1). Use a weighted error estimate.
                # The dominant error is in y1 (the small intermediate).
                # A reasonable relative error estimate:
                # rel_err ~ atol / max(|y|, atol) for each component, take max
                # But we don't know the true error. Use nfev as proxy for work.
                # Score = -log10(rel_err) - 0.5*log10(nfev)
                # We estimate rel_err from the tolerance: roughly atol/typical_scale
                # For Robertson: y0~1, y1~1e-5, y2~1
                # The error in y1 matters most for the "relative" error if we
                # consider the component with smallest magnitude.
                
                # Better approach: just track the result and prefer tighter tolerances
                # if time allows, since tighter tol => lower rel_err
                if best_result is None:
                    best_result = y_end
                    best_score = 0.0
                else:
                    # Prefer the tighter tolerance result (lower index in configs means looser)
                    # Since we go from loose to tight, later results are more accurate
                    best_result = y_end
                    best_score = 0.0
                    
        except Exception as e:
            break
    
    # If no successful solve, try a fallback with looser tolerance
    if best_result is None:
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=jac,
                rtol=1e-6,
                atol=1e-10,
                max_step=1.0,
            )
            if sol.success:
                best_result = sol.y[:, -1]
        except Exception:
            # Last resort: use a simple explicit method with many steps
            h = (t1 - t0) / 10000
            t = t0
            y = y0.copy()
            for _ in range(10000):
                y = y + h * f(t, y)
                t += h
            best_result = y
    
    return best_result
# EVOLVE-BLOCK-END
