"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate the fixed Lotka-Volterra IVP from t_span[0] to t_span[1] using a
    high-order explicit Runge-Kutta method (Dormand-Prince 8(7,5) variant or
    a simple high-order RK4 with adaptive step control).
    
    Since the problem is non-stiff and smooth, a fixed-step high-order method
    with a moderate number of steps should give high accuracy with few calls.
    
    We use a classic 4th-order Runge-Kutta with a large number of steps to ensure
    accuracy, but we can do better with a higher-order method. Let's use a
    fixed-step RK4 with enough steps to get rel error < 1e-14.
    
    Actually, let's use an adaptive RK45 (Dormand-Prince) to minimize calls while
    maintaining accuracy. But to keep it simple and robust, let's use a fixed
    step RK4 with a step size small enough that the global error is < 1e-14.
    
    The Lotka-Volterra system with these parameters is periodic and smooth.
    The reference solution uses DOP853 with rtol=1e-11, atol=1e-13.
    
    Let's use a fixed-step RK4 with h = 0.005 (1000 steps over [0,5]).
    That's 4000 calls. Let's check if that's enough.
    
    Actually, let's use a higher-order method. Let's implement a simple
    fixed-step RK4 but with more steps, or use an adaptive method.
    
    Let me implement an adaptive RK45 (Dormand-Prince) to minimize the number
    of function calls while achieving the required accuracy.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Adaptive RK45 (Dormand-Prince)
    # Coefficients for DP45
    a2 = 1/5
    a3 = 3/10
    a4 = 4/5
    a5 = 8/9
    a6 = 1
    a7 = 1
    
    b1 = 71/57600
    b2 = 0
    b3 = 71/16695
    b4 = 71/1920
    b5 = -212/7290
    b6 = 1/60
    b7 = 0
    
    c1 = 71/57600
    c2 = 0
    c3 = 71/16695
    c4 = 71/1920
    c5 = -212/7290
    c6 = 1/60
    c7 = 0
    
    d1 = 1/360
    d2 = 0
    d3 = 12845/32256
    d4 = -2197/7776
    d5 = 1/18
    d6 = 1/225
    d7 = 0
    
    # But actually, let's just use a fixed-step RK4 with enough steps.
    # For the Lotka-Volterra system, the dynamics are smooth and periodic.
    # Let's use h = 0.0025 (2000 steps), giving 8000 calls.
    # Or we can be smarter.
    
    # Let's use adaptive RK45 with tight tolerances.
    rtol = 1e-12
    atol = 1e-14
    
    h = (t1 - t0) / 100  # initial step
    t = t0
    total_calls = 0
    
    while t < t1:
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # RK45 stages
        k1 = f(t, y)
        total_calls += 1
        k2 = f(t + a2*h, y + h*a2*k1)
        total_calls += 1
        k3 = f(t + a3*h, y + h*(3/10*k1 - 9/40*k2))
        total_calls += 1
        k4 = f(t + a4*h, y + h*(4448/5529*k1 - 77/180*k2 + 32/9*k3))
        total_calls += 1
        k5 = f(t + a5*h, y + h*(19372/6561*k1 - 2536/1771*k2 + 64448/405*k3 - 212/729*k4))
        total_calls += 1
        k6 = f(t + a6*h, y + h*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))
        total_calls += 1
        
        # 5th order solution
        y_new = y + h*(35/384*k1 + 0*k2 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
        
        # Error estimate (difference between 4th and 5th order)
        err = h*(d1*k1 + d2*k2 + d3*k3 + d4*k4 + d5*k5 + d6*k6)
        err_norm = np.linalg.norm(err / (atol + rtol*np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            t = t + h
            y = y_new
            # Adjust step size
            if err_norm > 0:
                h_new = 0.9 * h * (1.0/err_norm)**(1/5.0)
            else:
                h_new = 2.0 * h
            h = min(h_new, 0.5*(t1 - t)) if t1 - t > 0 else h
        else:
            # Reject step, reduce h
            h = max(0.9 * h * (1.0/err_norm)**(1/5.0), 1e-12)
        
        if not np.all(np.isfinite(y)):
            # Fall back to fixed-step RK4
            break
    
    if not np.all(np.isfinite(y)) or t < t1 - 1e-15:
        # Fall back to fixed-step RK4 with many steps
        t = t0
        y = np.array(y0, dtype=float)
        steps = 10000
        h = (t1 - t0) / steps
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
            t += h
    
    return y

# EVOLVE-BLOCK-END
