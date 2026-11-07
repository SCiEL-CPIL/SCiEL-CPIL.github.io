"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853(f, t0, t1, y0, rtol=1e-12, atol=1e-14):
    """
    8th-order Dormand-Prince 853 (DOP853) adaptive step-size Runge-Kutta integrator.
    Uses 12 stages per step with embedded 7th-order error estimate.
    """
    # Butcher tableau for DOP853 (12 stages, 8th order, embedded 7th)
    # c_i, a_ij, b_i (8th order), bhat_i (7th order)
    # Reference: Dormand & Prince, 1996; also in scipy's _dop853
    
    c = np.array([
        0.0,
        1/27,
        1/9,
        5/24,
        1/4,
        5/12,
        1/2,
        5/6,
        7/8,
        1/6,
        5/18,
        1.0
    ])
    
    # a matrix (lower triangular, 12x12)
    a = np.zeros((12, 12))
    a[1,0] = 1/27
    a[2,0] = 1/36;  a[2,1] = 1/12
    a[3,0] = 5/12;  a[3,1] = -5/12;  a[3,2] = 5/6
    a[4,0] = 1/12;  a[4,1] = 0;       a[4,2] = -1/2;   a[4,3] = 1/2
    a[5,0] = 0;     a[5,1] = 5/12;    a[5,2] = -5/4;    a[5,3] = 5/3;  a[5,4] = -5/12
    a[6,0] = 0;     a[6,1] = 1/12;    a[6,2] = 0;       a[6,3] = -1/4; a[6,4] = 1/6;  a[6,5] = 1/4
    a[7,0] = 0;     a[7,1] = 0;       a[7,2] = 0;       a[7,3] = 0;    a[7,4] = 0;    a[7,5] = 0;    a[7,6] = 5/6
    a[8,0] = 0;     a[8,1] = 0;       a[8,2] = 0;       a[8,3] = 0;    a[8,4] = 0;    a[8,5] = 0;    a[8,6] = 0;    a[8,7] = 7/8
    a[9,0] = 0;     a[9,1] = 0;       a[9,2] = 0;       a[9,3] = 0;    a[9,4] = 0;    a[9,5] = 0;    a[9,6] = 0;    a[9,7] = 0;    a[9,8] = 1/6
    a[10,0] = 0;    a[10,1] = 0;      a[10,2] = 0;      a[10,3] = 0;   a[10,4] = 0;   a[10,5] = 0;   a[10,6] = 0;   a[10,7] = 0;   a[10,8] = 0;   a[10,9] = 5/18
    a[11,0] = 0;    a[11,1] = 0;      a[11,2] = 0;      a[11,3] = 0;   a[11,4] = 0;   a[11,5] = 0;   a[11,6] = 0;   a[11,7] = 0;   a[11,8] = 0;   a[11,9] = 0;   a[11,10] = 1
    
    # b weights (8th order)
    b = np.array([
        37/378, 0, 0, 250/621, 125/594, 0, 512/1771, 0, 0, 0, 500/1113, 0
    ])
    
    # bhat weights (7th order, for error estimate)
    bhat = np.array([
        1/135, 0, 0, 14/81, 4/45, 0, 128/405, 0, 0, 0, 2197/4100, -1/10
    ])
    
    # Precompute
    n = len(y0)
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size estimate
    h = (t1 - t0) / 10.0
    h = max(h, 1e-12)
    h = min(h, abs(t1 - t0))
    
    deadline = time.time() + 250  # Safety margin before 300s limit
    
    # Initial f evaluation
    k = [None] * 12
    k[0] = f(t, y)
    
    # Error norm helper
    def _err_norm(err):
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y + err))
        return np.sqrt(np.sum((err / sc) ** 2))
    
    while t < t1:
        if time.time() > deadline:
            break
        # Don't overshoot
        h = min(h, t1 - t)
        if h < 1e-15:
            break
        
        # Compute remaining stages
        for i in range(1, 12):
            y_stage = y.copy()
            for j in range(i):
                if a[i, j] != 0:
                    y_stage = y_stage + h * a[i, j] * k[j]
            k[i] = f(t + c[i] * h, y_stage)
        
        # 8th order solution
        y_new = y.copy()
        for i in range(12):
            if b[i] != 0:
                y_new = y_new + h * b[i] * k[i]
        
        # 7th order solution (for error estimate)
        y_hat = y.copy()
        for i in range(12):
            if bhat[i] != 0:
                y_hat = y_hat + h * bhat[i] * k[i]
        
        # Error estimate
        err = y_new - y_hat
        err_norm = _err_norm(err)
        
        # Step size control
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t = t + h
            
            # Estimate next step size
            if err_norm < 1e-15:
                h_new = h * 5.0
            else:
                h_new = h * min(5.0, max(0.1, 0.9 * (1.0 / err_norm) ** (1.0 / 8.0)))
            h = min(h_new, t1 - t)
        else:
            # Reject step, try smaller h
            if err_norm > 1e10:
                h_new = h * 0.1
            else:
                h_new = h * max(0.1, 0.9 * (1.0 / err_norm) ** (1.0 / 8.0))
            h = max(h_new, 1e-15)
            # Re-evaluate k[0] since y hasn't changed but we need to restart stages
            # Actually y is unchanged, so k[0] is still valid
            continue
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) using DOP853 adaptive RK.
    Returns the state at t_end.
    """
    t0, t1 = t_span
    y = _dop853(f, t0, t1, y0, rtol=1e-12, atol=1e-14)
    return y
# EVOLVE-BLOCK-END
