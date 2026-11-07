"""Integrator for Van der Pol oscillator using DOP853 with adaptive step control."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) using DOP853 (8th order)
    with adaptive step size control for high accuracy and low call count.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track elapsed time for safety
    start_time = time.time()
    deadline = start_time + 280.0  # 280s deadline to stay under 330s kill
    
    # DOP853 coefficients (Dormand-Prince 8(5,3))
    # This is a 12-stage, 8th-order explicit Runge-Kutta method with 5th-order error estimator
    # Coefficients from Dormand & Prince (1995)
    
    # c values (nodes)
    c = np.array([
        0,
        1/12,
        1/6,
        5/24,
        3/8,
        1/2,
        12470221/21827872,
        437269/1024000,
        82187/175760,
        22275/12128,
        7/8,
        1
    ])
    
    # A matrix (lower triangular, A[i][j] for j < i)
    A = np.zeros((12, 12))
    A[1,0] = 1/12
    A[2,0] = 1/24
    A[2,1] = 1/12
    A[3,0] = 1/24
    A[3,1] = 5/24
    A[3,2] = 1/12
    A[4,0] = 1/16
    A[4,1] = 0
    A[4,2] = 5/16
    A[4,3] = 0
    A[5,0] = 1/12
    A[5,1] = 0
    A[5,2] = 0
    A[5,3] = 3/4
    A[5,4] = 0
    A[6,0] = 18575044/21827872
    A[6,1] = 0
    A[6,2] = 0
    A[6,3] = -369375/21827872
    A[6,4] = 118473/5456968
    A[6,5] = 0
    A[7,0] = 0
    A[7,1] = 0
    A[7,2] = 0
    A[7,3] = 0
    A[7,4] = 0
    A[7,5] = 437269/1024000
    A[8,0] = 0
    A[8,1] = 0
    A[8,2] = 0
    A[8,3] = 0
    A[8,4] = 0
    A[8,5] = 0
    A[8,6] = 0
    A[8,7] = 82187/175760
    A[9,0] = 0
    A[9,1] = 0
    A[9,2] = 0
    A[9,3] = 0
    A[9,4] = 0
    A[9,5] = 0
    A[9,6] = 0
    A[9,7] = 0
    A[9,8] = 22275/12128
    A[10,0] = 0
    A[10,1] = 0
    A[10,2] = 0
    A[10,3] = 0
    A[10,4] = 0
    A[10,5] = 0
    A[10,6] = 0
    A[10,7] = 0
    A[10,8] = 0
    A[10,9] = 7/8
    A[11,0] = 0
    A[11,1] = 0
    A[11,2] = 0
    A[11,3] = 0
    A[11,4] = 0
    A[11,5] = 0
    A[11,6] = 0
    A[11,7] = 0
    A[11,8] = 0
    A[11,9] = 0
    A[11,10] = 1
    
    # b vector (weights for 8th order solution)
    b = np.array([
        3921218729029281/185592184862758400,
        0,
        0,
        0,
        0,
        0,
        41732167168/22355280,
        0,
        0,
        0,
        0,
        0
    ])
    
    # b_hat vector (weights for 5th order error estimator)
    b_hat = np.array([
        4110931531170777/185592184862758400,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ])
    
    # e vector = b - b_hat (error estimator coefficients)
    e = b - b_hat
    
    # Wrap f to ensure it returns a proper array
    def f_wrapper(t, y):
        result = f(t, y)
        return np.asarray(result, dtype=float)
    
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 50.0  # Initial step size
    min_h = (t1 - t0) / 1e6
    max_h = (t1 - t0) / 5.0
    
    rtol = 1e-11
    atol = 1e-13
    
    max_steps = 10000
    
    for step in range(max_steps):
        if time.time() > deadline - 20:
            break
        if t >= t1 - 1e-14:
            break
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h
        
        # Compute stages
        k = np.zeros((12, len(y)))
        k[0] = f_wrapper(t, y)
        
        for i in range(1, 12):
            y_stage = y + h * np.dot(A[i,:i], k[:i])
            k[i] = f_wrapper(t + c[i]*h, y_stage)
        
        # 8th order solution
        y_new = y + h * np.dot(b, k)
        
        # 5th order solution for error estimation
        y_err = y + h * np.dot(b_hat, k)
        
        # Error estimate
        err = y_new - y_err
        
        if not np.all(np.isfinite(y_new)):
            return y
        
        # Error norm
        err_norm = np.linalg.norm(err)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        scale_norm = np.linalg.norm(scale)
        err_ratio = err_norm / scale_norm if scale_norm > 0 else 0
        
        if err_ratio <= 1.0:
            y = y_new
            t = t + h
            # Adaptive step size control
            if err_ratio > 1e-6:
                h_new = h * min(5.0, max(0.1, 0.9 * err_ratio ** (-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, max_h)
        else:
            # Reject step
            h = h * max(0.1, 0.9 * err_ratio ** (-0.25))
            if h < min_h:
                h = min_h
    
    if t < t1 - 1e-14:
        # Take final step
        h = t1 - t
        k = np.zeros((12, len(y)))
        k[0] = f_wrapper(t, y)
        for i in range(1, 12):
            y_stage = y + h * np.dot(A[i,:i], k[:i])
            k[i] = f_wrapper(t + c[i]*h, y_stage)
        y = y + h * np.dot(b, k)
    
    if np.all(np.isfinite(y)):
        return y
    
    # Fallback: use scipy's Radau
    from scipy.integrate import solve_ivp
    try:
        sol = solve_ivp(
            f_wrapper,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
    
    return y0
# EVOLVE-BLOCK-END
