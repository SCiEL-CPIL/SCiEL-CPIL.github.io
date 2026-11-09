"""ODE integrator for the Lotka-Volterra system."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _dop853_coefficients():
    """
    Return the coefficients for the DOP853 method (8th order accurate, 7th order error estimator).
    These are the same coefficients used by scipy.integrate.solve_ivp with method='DOP853'.
    """
    # Nodes c (for 13-stage method)
    c = np.array([
        0.0,
        7.0 / 243.0,
        35.0 / 324.0,
        1237.0 / 4212.0,
        1537.0 / 4374.0,
        2009.0 / 7290.0,
        59.0 / 135.0,
        10.0 / 21.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ])
    
    # Butcher tableau A (lower triangular, 13x13)
    A = np.zeros((13, 13))
    A[1, 0] = 7.0 / 243.0
    A[2, 0] = 32.0 / 9.0
    A[2, 1] = -91.0 / 918.0
    A[3, 0] = 172531.0 / 2197000.0
    A[3, 1] = -72006.0 / 219700.0
    A[3, 2] = 142503.0 / 109850.0
    A[4, 0] = 439202.0 / 2197000.0
    A[4, 1] = -8.0
    A[4, 2] = 3680.0 / 51.0
    A[4, 3] = -8450.0 / 273.0
    A[5, 0] = -8.0 / 27.0
    A[5, 1] = 2.0
    A[5, 2] = -3544.0 / 273.0
    A[5, 3] = 1859.0 / 260.0
    A[5, 4] = -137.0 / 104.0
    A[6, 0] = 1857.0 / 4550.0
    A[6, 1] = -3683.0 / 4095.0
    A[6, 2] = 1341.0 / 364.0
    A[6, 3] = -41.0 / 182.0
    A[6, 4] = 8.0 / 105.0
    A[6, 5] = -2.0 / 95.0
    A[7, 0] = 1351237795.0 / 4751598450.0
    A[7, 1] = -8988234285.0 / 3678119250.0
    A[7, 2] = 13411270222.0 / 1839059625.0
    A[7, 3] = -44525303243.0 / 1839059625.0
    A[7, 4] = 10938873894.0 / 1839059625.0
    A[7, 5] = 2147319014.0 / 1839059625.0
    A[7, 6] = 1970144.0 / 129279.0
    A[8, 0] = -1772038.0 / 1606281.0
    A[8, 1] = 113.0 / 18.0
    A[8, 2] = -23450.0 / 6201.0
    A[8, 3] = 6340135.0 / 1326528.0
    A[8, 4] = -2140609.0 / 1326528.0
    A[8, 5] = 347.0 / 1458.0
    A[8, 6] = 11.0 / 84.0
    A[8, 7] = -7.0 / 81.0
    A[9, 0] = 19372.0 / 6561.0
    A[9, 1] = -398.0 / 27.0
    A[9, 2] = 2180.0 / 819.0
    A[9, 3] = 6821191.0 / 1651704.0
    A[9, 4] = -2425355.0 / 1651704.0
    A[9, 5] = 10604.0 / 1458.0
    A[9, 6] = -11.0 / 40.0
    A[9, 7] = 1.0 / 20.0
    A[9, 8] = 2.0 / 27.0
    A[10, 0] = 9017.0 / 3168.0
    A[10, 1] = -355.0 / 24.0
    A[10, 2] = 1375.0 / 1224.0
    A[10, 3] = 41161679.0 / 9350520.0
    A[10, 4] = -17646079.0 / 9350520.0
    A[10, 5] = 1130.0 / 729.0
    A[10, 6] = -23.0 / 60.0
    A[10, 7] = 1.0 / 20.0
    A[10, 8] = 1.0 / 9.0
    A[10, 9] = 1.0 / 24.0
    A[11, 0] = 59519.0 / 18750.0
    A[11, 1] = -2.0
    A[11, 2] = 134.0 / 625.0
    A[11, 3] = 2706098.0 / 935055.0
    A[11, 4] = -1170278.0 / 935055.0
    A[11, 5] = 1048.0 / 2025.0
    A[11, 6] = -1.0 / 4.0
    A[11, 7] = 1.0 / 12.0
    A[11, 8] = 1.0 / 6.0
    A[11, 9] = 1.0 / 12.0
    A[11, 10] = 1.0 / 30.0
    A[12, 0] = 46978.0 / 65625.0
    A[12, 1] = 56.0 / 225.0
    A[12, 2] = -19378.0 / 65625.0
    A[12, 3] = 10831952.0 / 935055.0
    A[12, 4] = -4617682.0 / 935055.0
    A[12, 5] = 1048.0 / 2025.0
    A[12, 6] = -1.0 / 4.0
    A[12, 7] = 1.0 / 12.0
    A[12, 8] = 1.0 / 6.0
    A[12, 9] = 1.0 / 12.0
    A[12, 10] = 1.0 / 30.0
    A[12, 11] = 1.0 / 30.0
    
    # 8th order weights b (for the 8th order solution)
    b8 = np.array([
        3025.0 / 13824.0,
        0.0,
        5923.0 / 15552.0,
        4400.0 / 1521.0,
        -20317.0 / 7290.0,
        2825.0 / 2187.0,
        -61001.0 / 33075.0,
        -103.0 / 220.0,
        256319.0 / 843900.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])
    
    # 7th order weights b7 (for the 7th order error estimator)
    b7 = np.array([
        6025.0 / 13824.0,
        0.0,
        5969.0 / 15552.0,
        4400.0 / 1521.0,
        -20317.0 / 7290.0,
        2825.0 / 2187.0,
        -61001.0 / 33075.0,
        -103.0 / 220.0,
        256319.0 / 843900.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])
    
    # Error weights e = b8 - b7
    e = b8 - b7
    
    return c, A, b8, e


def _dop853_step(f, t, y, h, c, A, b8, e):
    """
    Perform one DOP853 step. Returns y_new (8th order) and y_err (7th order error estimate).
    """
    n = len(y)
    K = np.empty((13, n))
    K[0] = f(t, y)
    
    for i in range(1, 13):
        y_stage = y.copy()
        for j in range(i):
            y_stage += h * A[i, j] * K[j]
        K[i] = f(t + h * c[i], y_stage)
    
    y_new = y.copy()
    for i in range(13):
        y_new += h * b8[i] * K[i]
    
    y_err = np.zeros(n)
    for i in range(13):
        y_err += h * e[i] * K[i]
    
    return y_new, y_err


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using an adaptive DOP853 method with very tight tolerances.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Get DOP853 coefficients
    c, A, b8, e = _dop853_coefficients()
    
    # Tight tolerances to drive error below the 1e-14 floor
    rtol = 1e-13
    atol = 1e-15
    
    # Initial step size estimate
    h = (t1 - t0) / 100
    if h <= 0:
        return y
    
    t = t0
    t_end = t1
    
    # Safety margins
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit with 20s margin
    
    # Main integration loop
    while t < t_end:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Don't overshoot t_end
        if t + h > t_end:
            h = t_end - t
        
        # Perform one DOP853 step
        y_new, y_err = _dop853_step(f, t, y, h, c, A, b8, e)
        
        # Check for finite values
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            if h < 1e-15:
                break
            continue
        
        # Compute error norm
        scale = np.maximum(rtol * np.maximum(np.abs(y), np.abs(y_new)), atol)
        err_norm = np.linalg.norm(y_err / scale) / np.sqrt(n)
        
        # Accept or reject step
        if err_norm <= 1.0:
            t += h
            y = y_new
            
            # Adapt step size
            if err_norm > 1e-15:
                h_new = h * safety * err_norm ** (-1.0 / 8.0)
            else:
                h_new = h * max_factor
            
            h = np.clip(h_new, min_factor * h, max_factor * h)
        else:
            # Reject step and reduce h
            h_new = h * safety * err_norm ** (-1.0 / 8.0)
            h = np.clip(h_new, min_factor * h, max_factor * h)
            if h < 1e-15:
                break
    
    return y
# EVOLVE-BLOCK-END
