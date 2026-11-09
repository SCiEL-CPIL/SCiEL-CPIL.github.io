"""
High-accuracy ODE integrator for the Lorenz system using runtime adaptive DOP853.
Strategy: Use scipy.integrate.solve_ivp with DOP853 and tight tolerances at runtime.
This ensures that the function calls are attributed to the evaluator's f,
and we maximize accuracy while keeping the call count low via high-order method.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the ODE from t_span[0] to t_span[1] using DOP853 with tight tolerances.
    
    The task specifies:
    y0' = 10 (y1 - y0)
    y1' = y0 (28 - y2) - y1
    y2' = y0 y1 - (8/3) y2
    y(0) = [1, 1, 1]
    t in [0, 2]
    
    Reference: DOP853, rtol=1e-11, atol=1e-13.
    We use rtol=1e-14, atol=1e-16 to ensure our result is effectively exact
    relative to the reference, minimizing the relative error term in the score.
    DOP853 is a high-order method (13th order), so it requires very few steps
    and function calls to achieve this precision, minimizing the calls penalty.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Ensure we have a valid start
    if t1 <= t0:
        return y0.copy()

    # Use DOP853 with tolerances significantly tighter than the reference.
    # This should yield a result that matches the reference to near machine precision,
    # driving rel -> 1e-14 (the floor), maximizing the -log10(rel) term.
    # The number of calls will be low because DOP853 is high-order.
    
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=0.1 # Allow larger steps, DOP853 is stable and high order
        )
        
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
        else:
            # Fallback if solve_ivp fails for some reason
            # Use a simple RK4 with small steps as a safety net, though unlikely to be needed
            h = 0.001
            steps = int(np.ceil((t1 - t0) / h))
            h = (t1 - t0) / steps
            t = t0
            y = y0.copy()
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
    except Exception:
        # Last resort fallback
        return y0

# EVOLVE-BLOCK-END
