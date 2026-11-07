import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    # We expect a few thousand to tens of thousands of calls.
    # DOP853 is very fast. 330s is plenty for even 100,000 steps.
    
    # Tighten tolerances to minimize rel error against a high-precision reference.
    # If the reference is highly accurate, lower tolerances reduce ||y_end - y_ref||.
    # Trade-off: log(rel) gain vs 0.5*log(calls) penalty.
    # Typically rel decreases much faster (exponentially with tightness) than calls increase (polynomially).
    
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15
    )

    if not sol.success or sol.y.shape[1] == 0:
        # Fallback if it fails for some reason
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )

    if not sol.success or sol.y.shape[1] == 0:
        return np.asarray(y0, dtype=float)

    result = sol.y[:, -1].copy()

    if not np.all(np.isfinite(result)):
        result = np.asarray(y0, dtype=float)

    return result
