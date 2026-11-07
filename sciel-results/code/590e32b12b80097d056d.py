import numpy as np

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using a custom fixed-step Dormand-Prince 5(4) integrator.
    This bypasses scipy's adaptive overhead and provides a predictable, low call count.
    """
    t_start, t_end = t_span
    y = np.array(y0, dtype=float)
    
    # Fixed step count
    # DP54 is 5th order.
    # N=200 steps should provide good accuracy.
    n_steps = 200
    h = (t_end - t_start) / n_steps
    
    # Dormand-Prince 5(4) Coefficients
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    
    for _ in range(n_steps):
        k = np.zeros((7, y.shape[0]))
        for j in range(7):
            k[j] = f(t_start + c[j]*h, y + h * (A[j] @ k[:j]))
        y = y + h * (b @ k)
        t_start += h
        
    return y
