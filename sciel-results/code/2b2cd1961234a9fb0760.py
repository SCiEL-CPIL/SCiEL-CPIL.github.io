"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Adaptive step-size Dormand-Prince 5(4) Runge-Kutta integrator.
    Returns the state at t_end.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    rtol = 1e-13
    atol = 1e-15
    
    # Dormand-Prince 5(4) coefficients
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    A = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [35/384, 0, 500/1113, 125/192, -2187/6784],
        [71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525],
    ]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b4 = [71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]
    
    h = (t1 - t0) / 50
    t = t0
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    max_iter = 100000
    iter_count = 0
    
    while t < t1 and iter_count < max_iter:
        iter_count += 1
        if t + h > t1:
            h = t1 - t
        if h < 1e-14:
            break
        
        # Compute stages
        k = []
        for i in range(6):
            yi = y.copy()
            for j in range(i):
                yi += h * A[i][j] * k[j]
            k.append(f(t + c[i+1] * h, yi))
        
        # 5th order solution
        y_new = y.copy()
        for i in range(6):
            y_new += h * b5[i] * k[i]
        
        # 4th order solution
        y_old = y.copy()
        for i in range(6):
            y_old += h * b4[i] * k[i]
        
        # Error
        err = y_new - y_old
        
        if not np.all(np.isfinite(err)) or not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((err / scale) ** 2))
        
        if err_norm <= 1.0:
            t = t + h
            y = y_new
            if err_norm > 0:
                factor = safety * (err_norm ** (-0.2))
            else:
                factor = max_factor
            factor = max(min_factor, min(max_factor, factor))
            h *= factor
        else:
            factor = safety * (err_norm ** (-0.2))
            factor = max(min_factor, factor)
            h *= factor
    
    return y
# EVOLVE-BLOCK-END
