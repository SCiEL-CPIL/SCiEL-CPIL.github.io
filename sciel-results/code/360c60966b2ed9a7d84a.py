import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Dormand-Prince 5(4) coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    b4 = np.array([51/576, 0, 757/16695, 393/640, -92097/339200, 187/2100, 1/40])
    b5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    
    rtol = 1e-10
    atol = 1e-12
    h = (t1 - t0) / 100
    min_h = 1e-12
    max_h = (t1 - t0)
    
    calls = 0
    def f_counted(t, y):
        nonlocal calls
        calls += 1
        return f(t, y)
        
    while t < t1 - 1e-14:
        h = min(h, t1 - t)
        
        k = np.zeros((7, len(y)))
        k[0] = f_counted(t, y)
        for i in range(1, 7):
            y_stage = y + h * np.sum(a[i, :i] * k[:i], axis=0)
            k[i] = f_counted(t + c[i]*h, y_stage)
            
        y4 = y + h * np.sum(b4 * k, axis=0)
        y5 = y + h * np.sum(b5 * k, axis=0)
        
        err = np.max(np.abs(y5 - y4))
        scale = np.max(atol + rtol * np.abs(y4))
        
        if err <= scale:
            t += h
            y = y5
            if err < 1e-30:
                h *= 5
            else:
                h *= 0.9 * (scale / err) ** 0.2
            h = min(h, max_h)
            h = max(h, min_h)
        else:
            h *= 0.9 * (scale / err) ** 0.25
            h = max(h, min_h)
            
        if calls > 2000000:
            break
            
    return y
