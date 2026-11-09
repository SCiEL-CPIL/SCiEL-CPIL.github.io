# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float, copy=True)
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
    # Van der Pol specific derivatives
    # y0' = y1
    # y1' = 5(1 - y0^2)y1 - y0
    
    def get_derivatives(y):
        u = y[0]
        v = y[1]
        
        # d0
        d0 = np.array([u, v])
        
        # d1
        u1 = v
        v1 = 5.0 * (1.0 - u*u) * v - u
        d1 = np.array([u1, v1])
        
        # d2
        u2 = v1
        v2 = 5.0*v1 - 10.0*u*u1*v - 5.0*u*u*v1 - u1
        d2 = np.array([u2, v2])
        
        # d3
        u3 = v2
        v3 = 5.0*v2 - 5.0*u*u*v2 - 10.0*u1*u1*v - 10.0*u*u2*v - 20.0*u*u1*v1 - u2
        d3 = np.array([u3, v3])
        
        # d4
        u4 = v3
        v4 = 5.0*v3 - 5.0*u*u*v3 - 30.0*u*u1*v2 - 30.0*u1*u2*v - 30.0*u1*u1*v1 - 30.0*u*u2*v1 - 10.0*u*u3*v - u3
        d4 = np.array([u4, v4])
        
        return [d0, d1, d2, d3, d4]
    
    t = t0
    h = 0.01
    rtol = 1e-12
    atol = 1e-14
    fact = [1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0]
    
    while t < t1 - 1e-15:
        if time.time() > deadline:
            break
            
        h = min(h, t1 - t)
        
        d = get_derivatives(y)
        
        # Taylor 4
        y_high = y.copy()
        for i in range(1, 5):
            y_high += (h ** i) * fact[i] * d[i]
            
        # Taylor 3 for error estimate
        y_low = y.copy()
        for i in range(1, 4):
            y_low += (h ** i) * fact[i] * d[i]
        
        err_vec = y_high - y_low
        scale = np.maximum(atol, rtol * np.abs(y_high))
        err_scaled = np.linalg.norm(err_vec / scale)
        
        if err_scaled > 1.0:
            factor = 0.9 * (1.0 / err_scaled) ** 0.25
            h *= factor
            if h < 1e-12:
                h = 1e-12
            continue
        
        y = y_high
        t += h
        
        if err_scaled < 1e-10:
            factor = 10.0
        else:
            factor = 0.9 * (1.0 / err_scaled) ** 0.2
        h *= factor
        
    return y
# EVOLVE-BLOCK-END
