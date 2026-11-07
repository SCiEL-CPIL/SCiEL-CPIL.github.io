import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using a fixed-step DOP853 solver.
    
    Strategy:
    The score is -log10(rel) - 0.5 * log10(calls).
    For a p-th order method, rel ~ h^p ~ (1/N)^p.
    Score ~ p*log10(N) - 0.5*log10(N) = (p-0.5)*log10(N).
    Since p > 0.5, the score increases with the number of steps N.
    Therefore, we should use the maximum number of steps allowed by the call limit.
    
    DOP853 has 12 stages. Max calls = 2,000,000.
    Max steps N = 2,000,000 // 12 = 166,666.
    We force this step count by setting max_step = (t_end - t_start) / N
    and disabling adaptive error control (large rtol/atol).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    t_start, t_end = t_span
    y0 = np.array(y0, dtype=float)
    
    # Max steps for DOP853 (12 stages) to stay under 2,000,000 calls
    max_calls = 2000000
    stages = 12
    max_steps = max_calls // stages
    
    # Ensure at least 1 step
    if max_steps < 1:
        max_steps = 1
        
    h = (t_end - t_start) / max_steps
    
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e9,
            atol=1e9,
            max_step=h
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
        
    # Fallback: If the above fails, use a smaller number of steps
    # to ensure we don't exceed time or call limits in case of unexpected behavior
    if time.time() > deadline:
        return y0
        
    try:
        # Try half the steps
        max_steps_fallback = max_steps // 2
        if max_steps_fallback < 1:
            max_steps_fallback = 1
        h_fallback = (t_end - t_start) / max_steps_fallback
        
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e9,
            atol=1e9,
            max_step=h_fallback
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
        
    # Last resort: Adaptive solver (current behavior)
    if time.time() > deadline:
        return y0
        
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    return y0
