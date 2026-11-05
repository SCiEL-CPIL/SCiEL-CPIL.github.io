import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    call_count = 0
    start_time = time.time()
    max_time = 310.0
    max_calls = 1950000

    def f_wrapped(t, y):
        nonlocal call_count
        call_count += 1
        elapsed = time.time() - start_time
        if elapsed > max_time:
            raise TimeoutError("Time limit exceeded")
        if call_count > max_calls:
            raise OverflowError("Call limit exceeded")
        return f(t, y)

    # Fixed-step RK4
    # Choose N to balance error and cost.
    # For 100-point convection-diffusion, h=1e-3 (N=500) is likely stable and accurate.
    # Let's try N=1000 for safety on accuracy, cost is low.
    N = 1000
    h = (t1 - t0) / N
    y = y0.copy()
    t = t0
    
    for i in range(N):
        k1 = f_wrapped(t, y)
        k2 = f_wrapped(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f_wrapped(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f_wrapped(t + h, y + h * k3)
        
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + h
        
        if i % 100 == 0 and time.time() - start_time > max_time * 0.8:
             # If running out of time, break and return current best
             # But we are at step 100 of 1000, so we have done 10% of work.
             # Better to just let it run if it's fast.
             pass

    return y
