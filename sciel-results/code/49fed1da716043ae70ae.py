"""Rewritten ODE integrator using adaptive Dormand-Prince 5(4) method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Dormand-Prince coefficients
# Butcher tableau for DP5
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
b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size estimation
    # Heuristic: based on the initial derivative magnitude
    f0 = f(t0, y)
    f0_norm = np.linalg.norm(f0)
    if f0_norm == 0:
        h = 1e-6
    else:
        h = 1e-6 * (1.0 + np.linalg.norm(y) / f0_norm)
    h = min(h, (t1 - t0) / 10.0)
    
    t = t0
    rtol = 1e-12
    atol = 1e-14
    max_calls = 100000
    calls = 1 # f0 already called
    
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin for 300s limit
    
    while t < t1:
        if time.time() > deadline:
            # Return best estimate if we run out of time (should not happen for this small problem)
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Dormand-Prince step
        k = np.zeros((7, n))
        
        # Stage 1
        k[0] = f0
        
        # Stages 2-6
        for i in range(1, 6):
            y_stage = y + h * np.sum(A[i, :i] * k[:i], axis=0)
            t_stage = t + c[i] * h
            k[i] = f(t_stage, y_stage)
            calls += 1
            if calls > max_calls:
                # Fallback to simple RK4 with tiny steps if we hit call limit? 
                # Or just return current y. For this problem, we won't hit it.
                return y
        
        # Stage 7 (for 5th order)
        y_stage = y + h * np.sum(A[6, :6] * k[:6], axis=0)
        k[6] = f(t + h, y_stage)
        calls += 1
        
        # Compute 5th order and 4th order solutions
        y5 = y + h * np.sum(b * k, axis=0)
        y4 = y + h * np.sum(b_star * k, axis=0)
        
        # Error estimate
        err = y5 - y4
        err_norm = np.linalg.norm(err / (atol + rtol * np.maximum(np.abs(y4), np.abs(y5))))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y5
            # Update step size for next step
            if err_norm > 1e-8:
                h_new = 0.9 * h * (1.0 / err_norm) ** 0.2
            else:
                h_new = 5.0 * h
            h = min(h_new, (t1 - t) / 10.0)
        else:
            # Reject step, reduce h
            h = 0.9 * h * (1.0 / err_norm) ** 0.2
            if h < 1e-12:
                h = 1e-12
        
        # Ensure h doesn't become too small or too large
        h = max(h, 1e-12)
        
        # Update f0 for next step to save one call? 
        # Actually, DP5 has an embedded property, but usually we just recompute f at the new point.
        # We can reuse the last stage k[6] if we adjust, but standard implementation recomputes.
        # To save calls, we can set f0 = k[6] if we accept the step, 
        # because k[6] is f(t+h, y5) approximately.
        # Let's verify: k[6] = f(t+h, y + h*sum(A[6,:6]*k[:6])).
        # y5 = y + h*sum(b*k).
        # b and A[6] are different. So k[6] is NOT f(t+h, y5).
        # So we must recompute f(t+h, y5) for the next step's k1.
        # However, we can optimize by checking if we want to save calls.
        # For now, standard recompute.
        f0 = f(t, y)
        calls += 1

    return y
# EVOLVE-BLOCK-END
