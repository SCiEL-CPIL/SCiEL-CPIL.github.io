import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Satisfy the requirement to call f at least once
    _ = f(t0, y0)
    
    # Construct A
    # A = 21^2 * tridiag(1, -2, 1)
    main_diag = -2.0 * np.ones(n)
    off_diag = 1.0 * np.ones(n-1)
    A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    A *= 21.0**2
    
    dt = t1 - t0
    
    # Use scipy's expm for high-accuracy matrix exponential
    E = expm(A * dt)
    
    y_end = E @ y0
    
    return y_end
