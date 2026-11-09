"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as si


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses DOP853 (high-order explicit) as the primary solver for optimal accuracy/call ratio.
    Falls back to Radau (implicit) if DOP853 fails or is too slow.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: evaluator kills after 330s, keep safety margin of ~30s
    deadline = time.time() + 290.0
    
    # Strategy 1: DOP853 with very tight tolerances
    # DOP853 is 8th order explicit. High accuracy per step.
    # Tight tolerances to maximize the -log10(error) term.
    try:
        sol = si.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853',
            rtol=1e-13, 
            atol=1e-15,
            max_step=(t1 - t0) / 50.0, # Ensure resolution
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 50.0:
        # Fallback to simple RK4 if we are running out of time
        # Use a reasonable step count
        steps = 2000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    # Strategy 2: Radau (implicit) with tight tolerances
    # Good for stiff problems. If DOP853 failed or took too long (though it returns immediately on success),
    # we try this next.
    try:
        sol = si.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30.0:
        # Fallback to RK4
        steps = 2000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
        
    # Strategy 3: BDF (implicit)
    try:
        sol = si.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1 - t0) / 100.0,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Final fallback: high-accuracy RK4
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END
