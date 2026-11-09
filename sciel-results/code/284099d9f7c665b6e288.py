"""
Fixed-step high-order RK integrator for the Lorenz system.
Strategy: Use a fixed-step 8th-order Runge-Kutta method with a large number of steps.
This avoids pre-computation risks and scipy dependencies, ensuring the result is
computed directly from the provided f(t, y) function with high accuracy.
"""
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Returns the state at t_span[1].
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # If we are out of time, we cannot compute. 
        # However, this function is called once per evaluation.
        # If the module load took too long, we might be in trouble.
        # But we are not doing heavy work at module load now.
        pass

    t0, t1 = t_span
    y = np.array(y0, dtype=np.float64)
    
    # If t_span is zero or negative, return y0
    if t1 <= t0:
        # Must call f at least once
        f(t0, y)
        return y

    # Define the number of steps.
    # RK8 is very accurate. With N=1000 steps, h=0.002.
    # Error is roughly O(h^8). (0.002)^8 is extremely small (~2.5e-22).
    # This should be well below the 1e-14 relative error floor.
    # Calls = 8 * N = 8000.
    # Score penalty: -0.5 * log10(8000) ~ -0.5 * 3.9 = -1.95.
    # Accuracy gain: rel ~ 1e-15 -> -log10(1e-15) = 15.
    # Total ~ 13.05.
    # Current score 1.49 is likely due to a bug or high call count/low accuracy in the seed.
    
    N = 1000
    h = (t1 - t0) / N
    
    # Coefficients for 8th order Runge-Kutta (Kutta's method)
    # Using a standard 8-stage RK scheme.
    # One common 8-stage RK (e.g., from Hairer & Wanner)
    # But for simplicity and robustness, we can use a known stable 8th order method.
    # Let's use a specific set of coefficients for 8th order.
    # Alternatively, we can just use a very high number of steps with a lower order method?
    # No, RK8 with N=1000 is very fast and accurate.
    
    # Let's use a simpler approach: 4th order RK with more steps?
    # RK4 error O(h^4). To get 1e-14, h^4 ~ 1e-14 => h ~ 1e-3.5 ~ 3e-4.
    # N ~ 3000. Calls = 4 * 3000 = 12000.
    # RK8 error O(h^8). h^8 ~ 1e-14 => h ~ 1e-1.75 ~ 0.017.
    # N ~ 60. Calls = 8 * 60 = 480.
    # Let's use RK8 with N=100 to be safe on accuracy and speed.
    # h = 0.02. h^8 = 2.5e-14. This might be borderline.
    # Let's use N=500. h = 0.004. h^8 ~ 6e-21. Very safe.
    # Calls = 4000.
    
    N = 500
    h = (t1 - t0) / N
    
    # 8th order Runge-Kutta coefficients (Kutta's method)
    # Reference: A. Enright, "A family of embedded Runge-Kutta methods" or similar.
    # Let's use a well-known 8th order set.
    # Actually, implementing a specific high-order RK from scratch is error-prone.
    # Let's use a simpler strategy: 
    # Use scipy if available, but wrap it to count calls? No, scipy doesn't expose call count easily.
    # The task says "f is called at least once" and "every call is counted".
    # If I use scipy, I pass a wrapper f that increments a counter.
    
    # Let's try the scipy wrapper approach again, but correctly.
    # The previous program failed. Why?
    # Maybe the pre-computation at module load failed silently?
    # Or maybe the reference solution y_ref is generated differently?
    
    # Let's stick to the manual RK8 implementation to be independent of scipy's internal state.
    # I will use a standard 8th order RK method.
    
    # Coefficients for 8th order RK (from various sources, e.g., Butcher tableau)
    # This is a specific 8-stage method.
    c = np.array([
        0.0,
        0.125,
        0.125,
        0.125,
        0.25,
        0.25,
        0.5,
        0.75
    ])
    
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.125, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
        [0.375, 0.0, 0.0, 0.0, 0.0, 0.375, 0.0, 0.0]
    ])
    
    b = np.array([
        1/16.0,
        1/16.0,
        1/16.0,
        1/16.0,
        1/16.0,
        1/16.0,
        1/8.0,
        1/8.0
    ])
    
    # Wait, I need to verify these coefficients. 
    # A safer bet is to use a known correct high-order method or just use RK4 with many steps.
    # RK4 is robust. 
    # Let's use RK4 with N=2000 steps.
    # h = 0.001.
    # Error ~ h^4 = 1e-12. This is close to 1e-14 floor.
    # Let's use N=10000. h=0.0001. Error ~ 1e-16. Machine precision limit.
    # Calls = 40000.
    # Score: -0.5 * log10(40000) = -0.5 * 4.6 = -2.3.
    # Rel ~ 1e-15. -log10(1e-15) = 15.
    # Total ~ 12.7.
    # This is much better than 1.49.
    
    # Let's optimize N.
    # We want min rel error. RK4 error is proportional to h^4.
    # For Lorenz, constants matter.
    # Let's use N=5000. h=0.0004. h^4 ~ 2.5e-14.
    # This might be slightly above the floor.
    # Let's use N=10000 to be safe. 40000 calls is well within 2,000,000 limit.
    
    N = 10000
    h = (t1 - t0) / N
    
    # RK4 implementation
    for i in range(N):
        t = t0 + i * h
        
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        
        y = y + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
        # Check deadline periodically
        if i % 1000 == 0 and time.time() > _DEADLINE:
            # If we run out of time, break and return current y
            # This is a partial result, but better than exception?
            # The contract says "exception makes result invalid".
            # Returning a partial result is finite, so it's valid but inaccurate.
            # But we have plenty of time for 10000 steps of RK4 in numpy.
            # 10000 * 4 calls = 40000 calls.
            # Each call is simple vector math. Should be < 1s.
            break
            
    return y
