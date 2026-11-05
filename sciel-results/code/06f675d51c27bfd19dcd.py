import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin

    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Step 1: Determine if the system is linear and time-invariant.
    # We assume it is, based on the task description.
    # We need to find A and b such that f(t, y) = A y + b.
    
    # Get b
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    
    # Check time invariance? 
    # f(t1, 0) should equal b.
    # This costs 1 call.
    # If not equal, it's time-dependent.
    # But task says time-invariant.
    # I will skip the check to save calls, but if the result is bad, fallback.
    
    # Get A columns
    A = np.zeros((n, n))
    for i in range(n):
        if time.time() > deadline:
            break
        e_i = np.zeros(n)
        e_i[i] = 1.0
        col = f(t0, e_i)
        A[:, i] = col - b
        
    if time.time() > deadline:
        # Fallback to RK4 with few steps? No, return best effort.
        # If we broke early, A is incomplete.
        # Just return y0? Or run a quick RK4?
        # Let's run a very coarse RK4 as fallback.
        y = y0.copy()
        steps = 100
        h = dt / steps
        t = t0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    # Verify linearity with one more call?
    # f(t0, y0) should be A y0 + b.
    # f_check = f(t0, y0)
    # expected = A @ y0 + b
    # if not np.allclose(f_check, expected):
    #     # Fallback
    
    # Solve
    try:
        if np.allclose(b, 0, rtol=1e-10, atol=1e-12):
            M = expm(A * dt)
            y_end = M @ y0
        else:
            # Affine case
            # y(t) = expm(A dt) y0 + \int_0^dt expm(A (dt - s)) b ds
            # = expm(A dt) y0 + (expm(A dt) - I) A^{-1} b
            A_inv = np.linalg.inv(A)
            M = expm(A * dt)
            y_end = M @ y0 + (M - np.eye(n)) @ A_inv @ b
    except np.linalg.LinAlgError:
        # Fallback to RK4
        y = y0.copy()
        steps = 2000
        h = dt / steps
        t = t0
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        y_end = y
        
    return y_end
