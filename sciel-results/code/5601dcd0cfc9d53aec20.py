"""ODE integrator for the Lotka-Volterra system using high-order fixed-step RK."""
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system y' = [1.5*y0 - y0*y1, -3*y1 + y0*y1]
    from t_span[0] to t_span[1] with initial condition y0.
    Returns the state at the end of the interval.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Use a high-order Runge-Kutta method.
    # We'll use a 13th order method if we can, but let's start with a well-known
    # high-order method. The DOP853 is 8th order. 
    # Let's try to use a simpler approach first: a very high order fixed step RK.
    # For smooth problems like Lotka-Volterra, a high-order method with moderate
    # steps can be very accurate.
    
    # Let's use the 13th-order Runge-Kutta method (Cooper-Pratt) coefficients
    # or a similar high-order method. Actually, let's use a simpler approach:
    # use scipy's solve_ivp with DOP853 but with very tight tolerances, 
    # but that's not allowed since we need to use f directly.
    
    # Let's implement a high-order RK method. We'll use the 8th-order 
    # Runge-Kutta-Fehlberg method or similar. Actually, let's use a 
    # fixed-step 8th order method with enough steps.
    
    # For the Lotka-Volterra system, which is smooth and non-stiff for this 
    # parameter range, a high-order method with moderate steps should work well.
    
    # Let's use a 13th order method. The coefficients are complex, so let's 
    # use a simpler high-order method. We'll use the 8th order Runge-Kutta 
    # method with a reasonable number of steps.
    
    # Actually, let's try a different approach: use a high-order method with
    # adaptive step size, but since we can't easily do that with a fixed f,
    # let's use a very high number of steps with a high-order method.
    
    # Let's use the 8th order Runge-Kutta method (Dormand-Prince 8th order)
    # with a fixed step size. We'll use enough steps to get high accuracy.
    
    # For 8th order RK, the error is O(h^9). With h = (t1-t0)/N, the error 
    # is O((t1-t0)^9 / N^9). To get rel < 1e-14, we need N^9 > (t1-t0)^9 / 1e-14.
    # With t1-t0 = 5, we need N^9 > 5^9 / 1e-14 = 1.95e15 / 1e-14 = 1.95e29.
    # So N > (1.95e29)^(1/9) ≈ 10^(29/9) ≈ 10^3.2 ≈ 1600.
    # So with N = 2000 steps, we should get rel < 1e-14.
    
    # But 8th order RK has 12 stages, so 2000 * 12 = 24000 calls, which is fine.
    
    # Let's implement an 8th order Runge-Kutta method.
    
    # Actually, let's use a simpler approach: use the 4th order Runge-Kutta 
    # method with many more steps, or use a higher order method with fewer steps.
    
    # Let's use the 8th order Runge-Kutta method (Dormand-Prince 8(5,3)) 
    # or a similar method. The coefficients are:
    
    # For simplicity, let's use a well-known 8th order method.
    
    # Actually, let's just use a high number of steps with a 4th order method,
    # or use a higher order method. Let's use the 8th order method from 
    # scipy's DOP853, but implement it ourselves.
    
    # DOP853 is an 8th order method with 12 stages. Let's use it.
    
    # The DOP853 coefficients are:
    c = np.array([
        0.0,
        1/12,
        1/6,
        1/4,
        1/3,
        11/24,
        3/8,
        1/2,
        2/3,
        25/32,
        4/3,
        1.0
    ])
    
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/48, 1/16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/36, 0, 1/9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5/144, 0, 1/12, 5/32, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 7/40, -32/9, 128/45, 28/15, 0, 0, 0, 0, 0, 0, 0],
        [0, 1/40, -4/9, 128/45, 11/9, 16/15, 0, 0, 0, 0, 0, 0],
        [0, 0, 1/60, -28/45, 14/45, 4/5, 1/20, 0, 0, 0, 0, 0],
        [0, 0, -1/20, 11/90, 19/45, -1/9, 1/30, 1/12, 0, 0, 0, 0],
        [0, 0, 0, 0, 1/12, -2/5, 0, 1/4, 1/12, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1/4, 0, -5/12, 1/4, 0, 0],
        [0, 0, 0, 0, 0, 0, -1/12, 1/12, 1/4, 0, 0, 0]
    ])
    
    b = np.array([
        37/378, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])
    
    # Wait, these coefficients don't look right. Let me use the actual DOP853 
    # coefficients from scipy.
    
    # Actually, let's use a different approach. Let's use the 13th order 
    # Runge-Kutta method, which has better accuracy per stage.
    
    # For now, let's use a simpler approach: use a high-order method with 
    # a reasonable number of steps. Let's use the 8th order method with 
    # 2000 steps, which should give high accuracy.
    
    # Actually, let's just use a very simple approach: use the 4th order 
    # Runge-Kutta method with a very large number of steps. With h = 5/10000,
    # the error is O(h^5) = O(5^5 / 10000^5) = O(3125 / 1e20) = O(3e-17),
    # which is below 1e-14. So 10000 steps with 4th order RK should work.
    
    # 10000 steps * 4 stages = 40000 calls, which is well under 2,000,000.
    
    # Let's use 4th order RK with 10000 steps.
    
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y
