import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince 5(4) coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    a = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    b5 = np.array([1/35, 0, 9/28, 9/56, 32/91, 19/210])
    b4 = np.array([51/57200, 0, 3000/3185, 18000/3185, 185/1104, 55/572])
    e = b5 - b4
    
    rtol = 1e-12
    atol = 1e-14
    
    h = (t1 - t0) / 100.0
    t = t0
    calls = 0
    max_calls = 1500000 # Safety margin
    
    # Initial step size selection can be improved, but this is okay.
    
    while t < t1:
        if calls > max_calls:
            break
            
        if t + h > t1:
            h = t1 - t
        if h < 1e-12:
            h = 1e-12
            
        # Calculate stages k1..k6
        k = np.zeros((6, n))
        
        # k1
        k[0] = f(t, y)
        calls += 1
        
        # k2
        y_temp = y + h * (a[1,0]*k[0])
        k[1] = f(t + c[1]*h, y_temp)
        calls += 1
        
        # k3
        y_temp = y + h * (a[2,0]*k[0] + a[2,1]*k[1])
        k[2] = f(t + c[2]*h, y_temp)
        calls += 1
        
        # k4
        y_temp = y + h * (a[3,0]*k[0] + a[3,1]*k[1] + a[3,2]*k[2])
        k[3] = f(t + c[3]*h, y_temp)
        calls += 1
        
        # k5
        y_temp = y + h * (a[4,0]*k[0] + a[4,1]*k[1] + a[4,2]*k[2] + a[4,3]*k[3])
        k[4] = f(t + c[4]*h, y_temp)
        calls += 1
        
        # k6
        y_temp = y + h * (a[5,0]*k[0] + a[5,1]*k[1] + a[5,2]*k[2] + a[5,3]*k[3] + a[5,4]*k[4])
        k[5] = f(t + c[5]*h, y_temp)
        calls += 1
        
        # Compute error
        err = np.zeros(n)
        for i in range(6):
            err += e[i] * k[i]
            
        # Compute norm for error control
        # err_norm / scale <= 1
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y + h * np.dot(b5, k)))
        err_vec = err / scale
        err_norm = np.linalg.norm(err_vec)
        
        if err_norm <= 1.0:
            # Accept step
            y = y + h * np.dot(b5, k)
            t = t + h
            
            # Adjust step size
            if err_norm > 0:
                factor = 0.9 * (err_norm ** (-0.2))
            else:
                factor = 5.0
            h = h * min(max(factor, 0.1), 5.0)
        else:
            # Reject step
            factor = 0.9 * (err_norm ** (-0.25))
            h = h * min(max(factor, 0.1), 5.0)
            
    return y
