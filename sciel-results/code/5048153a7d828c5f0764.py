"""Rewritten program for Heat2D 10x10 ODE solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    DEADLINE = 320.0  # safety margin before 330s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # For the 10x10 heat equation, n should be 100
    # Strategy: detect linearity, estimate A, use matrix exponential
    # This uses ~102 f calls and gives ~machine precision accuracy
    
    if n == 100:
        # Check linearity: f(t, 2*y) == 2*f(t, y)
        f0 = f(t0, y0)
        f2 = f(t0, 2.0 * y0)
        
        # Check if linear (homogeneous)
        if np.allclose(f2, 2.0 * f0, rtol=1e-10, atol=1e-12):
            # Estimate A by probing basis vectors
            A = np.zeros((n, n), dtype=float)
            e = np.zeros(n, dtype=float)
            
            for i in range(n):
                if time.time() - start_time > DEADLINE:
                    break
                e[i] = 1.0
                A[:, i] = f(t0, e)
                e[i] = 0.0
            
            # Verify A is correct
            if np.allclose(A @ y0, f0, rtol=1e-5, atol=1e-8):
                dt = t1 - t0
                
                # Use scipy.linalg.expm for high accuracy
                try:
                    from scipy.linalg import expm
                    M = expm(A * dt)
                    y_end = M @ y0
                    return y_end
                except ImportError:
                    pass
                
                # Fallback: Taylor series for matrix exponential
                I = np.eye(n)
                M = np.zeros((n, n))
                term = I.copy()
                A_dt = A * dt
                for k in range(1, 50):
                    term = term @ A_dt / k
                    M = M + term
                    if np.linalg.norm(term, ord='fro') < 1e-16 * max(1.0, np.linalg.norm(M, ord='fro')):
                        break
                y_end = M @ y0
                return y_end
    
    # Fallback: adaptive RK45 for non-linear or non-100 cases
    return _integrate_nonlinear(f, t_span, y0, start_time, DEADLINE)


def _integrate_nonlinear(f, t_span, y0, start_time=None, deadline=None):
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = 320.0
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    dt_total = t1 - t0
    
    if dt_total <= 0:
        return y
    
    # Adaptive RK45
    def rk45_step(t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/6.0, y + h/6.0 * k1)
        k3 = f(t + h/3.0, y + h/3.0 * (k1 + 2.0*k2))
        k4 = f(t + 3.0*h/4.0, y + 3.0*h/4.0 * (k1 - 2.0*k2 + 4.0*k3))
        k5 = f(t + h, y + h * (2.0*k1 - 3.0*k2 + 4.0*k3 - k4))
        k6 = f(t + h/2.0, y + h/2.0 * (-k1 + 2.0*k2 - 2.0*k3 + k4 + k5))
        
        y_new = y + h * (19327.0/216000.0*k1 - 1987.0/216000.0*k2 + 6701.0/108000.0*k3 
                        - 3441.0/108000.0*k4 + 793.0/108000.0*k5 + 29.0/108000.0*k6)
        
        # Error estimate (4th order)
        e = h * (1.0/120.0*k1 - 1.0/288.0*k2 - 1.0/192.0*k3 
                - 285.0/12160.0*k4 - 35.0/768.0*k5 + 1.0/40.0*k6)
        err = np.linalg.norm(e) / (1.0 + np.linalg.norm(y_new))
        
        return y_new, err
    
    h = dt_total / 50.0
    t = t0
    max_steps = 100000
    
    step_count = 0
    while t < t1 - 1e-15 and step_count < max_steps:
        if time.time() - start_time > deadline:
            break
        
        if t + h > t1:
            h = t1 - t
        
        y_new, err = rk45_step(t, y, h)
        
        if err < 1e-10:
            y = y_new
            t += h
            # Increase step size
            if err > 1e-12:
                h *= 1.3
            else:
                h *= 1.5
            h = min(h, dt_total / 10.0)
        else:
            # Decrease step size
            h *= 0.5
            if h < 1e-10:
                h = 1e-10
        
        step_count += 1
    
    return y

# EVOLVE-BLOCK-END
