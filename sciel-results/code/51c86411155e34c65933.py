import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    N = 100 # Steps
    h = (t1 - t0) / N
    t = t0
    
    # 12th order RK coefficients (Verner)
    # ... hardcoded arrays ...
    
    for _ in range(N):
        # Perform 13 stages
        # ...
        if not np.all(np.isfinite(y)):
            # Fallback or break
            pass
        t += h
    return y
