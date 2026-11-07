import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = y0.copy()
    t = t0
    
    # Initial step size
    h = (t1 - t0) / 10.0
    if h == 0:
        return y
    
    rtol = 1e-10
    atol = 1e-12
    
    calls = 0
    max_calls = 2000000

    def step(t, y, h):
        nonlocal calls
        k1 = f(t, y)
        calls += 1
        k2 = f(t + h/5.0, y + h*(k1/5.0))
        calls += 1
        k3 = f(t + 3.0*h/10.0, y + h*(3.0*k1/40.0 + 9.0*k2/40.0))
        calls += 1
        k4 = f(t + 4.0*h/5.0, y + h*(44.0*k1/45.0 - 56.0*k2/15.0 + 32.0*k3/9.0))
        calls += 1
        k5 = f(t + h, y + h*(19372.0*k1/6561.0 - 25360.0*k2/2187.0 + 64448.0*k3/6561.0 - 212.0*k4/729.0))
        calls += 1
        k6 = f(t + 7.0*h/8.0, y + h*(9017.0*k1/3168.0 - 355.0*k2/33.0 + 46732.0*k3/5247.0 + 49.0*k4/176.0 - 5103.0*k5/18656.0))
        calls += 1
        k7 = f(t + h, y + h*(35.0*k1/384.0 + 500.0*k3/1113.0 + 125.0*k4/192.0 - 2187.0*k5/6784.0 + 11.0*k6/84.0))
        calls += 1
        
        y_4 = y + h*(35.0*k1/384.0 + 500.0*k3/1113.0 + 125.0*k4/192.0 - 2187.0*k5/6784.0 + 11.0*k6/84.0)
        y_5 = y + h*(5179.0*k1/57600.0 + 7571.0*k3/16695.0 + 393.0*k4/640.0 - 92097.0*k5/339200.0 + 187.0*k6/2100.0 + k7/40.0)
        
        err = np.abs(y_5 - y_4)
        
        return y_5, err

    while t < t1:
        if calls > max_calls:
            break
            
        if t + h > t1:
            h = t1 - t
        
        y_new, err = step(t, y, h)
        
        scale = np.maximum(atol, rtol * np.abs(y_new))
        err_norm = np.max(err / scale)
        
        if err_norm <= 1.0:
            t += h
            y = y_new
            # Adjust h
            if err_norm > 0:
                h *= 0.9 * (1.0 / err_norm) ** (1.0 / 5.0)
            else:
                h *= 1.5
            h = min(h, 0.5 * (t1 - t))
        else:
            h *= 0.9 * (1.0 / err_norm) ** (1.0 / 5.0)
            h = max(h, 1e-10)
            
    return y
