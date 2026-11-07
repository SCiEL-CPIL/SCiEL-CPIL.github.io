"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_step(f, t, y, h):
    """Single RK45 step. Returns y_new, err_est, nfev."""
    a2 = 1/4
    a3 = 3/8
    a4 = 12/13
    a5 = 1
    a6 = 1/2

    b21 = 1/4
    b31 = 3/32
    b32 = 9/32
    b41 = 1932/2197
    b42 = -7200/2197
    b43 = 7296/2197
    b51 = 439/216
    b52 = -8
    b53 = 3680/513
    b54 = -845/4104
    b61 = -8/27
    b62 = 2
    b63 = -3544/2565
    b64 = 1859/4104
    b65 = -11/40

    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1

    k1 = f(t, y)
    k2 = f(t + c2*h, y + h*(b21*k1))
    k3 = f(t + c3*h, y + h*(b31*k1 + b32*k2))
    k4 = f(t + c4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    k5 = f(t + c5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    k6 = f(t + h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

    y_new = y + h*(361/50400*k1 + 300/4369*k2 + 180/13*k3 + 3600/13*k4 + 500/1113*k5 - 5*k6)
    # Error estimate: difference between 4th and 5th order
    err = h*(361/50400*k1 - 300/4369*k2 + 180/13*k3 - 3600/13*k4 + 500/1113*k5 + 5*k6)
    # Actually let me use the standard error formula for DOPRI5
    # e5 = y5 - y4 where y5 is 5th order, y4 is 4th order
    # y4 = y + h*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3 - 1859/13512*k4)
    # y5 = y + h*(25/216*k1 + 1408/2565*k2 + 2197/4104*k3 - 1/10*k4 + 1/12*k5)
    # err = y5 - y4
    err = h*(25/216*k1 + 1408/2565*k2 + 2197/4104*k3 - 1/10*k4 + 1/12*k5
             - 1932/2197*k1 + 7200/2197*k2 - 7296/2197*k3 + 1859/13512*k4)
    return y_new, err, 6


def integrate(f, t_span, y0):
    """Adaptive RK45 (DOPRI5) integrator for the Lorenz system."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 310.0  # 20s margin before 330s kill
    
    # Tolerances - aim for very high accuracy
    rtol = 1e-13
    atol = 1e-15
    
    # Initial step size
    h = (t1 - t0) / 100
    t = t0
    max_calls = 2000000
    calls = 0
    
    # Safety factor for step size adjustment
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    while t < t1:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Check call budget
        if calls > max_calls - 100:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Take a step
        y_new, err, nfev = _rk45_step(f, t, y, h)
        calls += nfev
        
        if not np.all(np.isfinite(y_new)):
            # Step failed, reduce h
            h *= 0.5
            if h < 1e-15:
                break
            continue
        
        # Error norm
        err_norm = np.max(np.abs(err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            
            # Adjust step size
            if err_norm < 1e-8:
                factor = max_factor
            else:
                factor = safety * (err_norm ** (-0.2))
            factor = max(min_factor, min(max_factor, factor))
            h *= factor
        else:
            # Reject step, reduce h
            factor = safety * (err_norm ** (-0.2))
            factor = max(min_factor, factor)
            h *= factor
    
    return y
# EVOLVE-BLOCK-END
