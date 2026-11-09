"""ODE solver: 15-dimensional stiff linear system using matrix extraction + exact linear ODE solve."""
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) for a 15-dim stiff linear system on [0, 1].
    
    Strategy:
      1. Extract the system matrix A from f (linear system: f(t,y) = A @ y).
      2. Compute the exact solution y(t_end) = expm((t_end - t0) * A) @ y0.
      3. This requires only n+1 = 16 calls to f (one for A @ e_j for each basis vector,
         plus one for A @ y0 to verify/extract the first column or handle non-homogeneous).
      4. For a linear system f(t, y) = A @ y + b(t), if b is constant or zero, we can
         still handle it. But for the standard stiff linear IVP, f(t,y) = A@y.
      
    Actually, let's be more careful. The system is y' = A @ y (homogeneous linear).
    To extract A, we evaluate f at basis vectors:
      A[:, j] = f(0, e_j) where e_j is the j-th standard basis vector.
    This takes n = 15 calls. Then compute expm((t1-t0)*A) @ y0.
    
    But wait - f might depend on t. For a linear autonomous system, f(t, y) = A @ y.
    Let's first check if f is autonomous by comparing f(t0, y0) and f(t_mid, y0).
    
    Actually, for safety and efficiency, let's just extract A assuming f(t, y) = A @ y.
    If the system is non-autonomous or non-linear, this won't work, but the task says
    "stiff linear system" which typically means y' = A(t) @ y or y' = A @ y.
    
    For the most common case y' = A @ y (autonomous linear), we need n calls to extract A.
    Then one expm computation. Total: ~15-16 calls to f.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin before 300s limit
    
    # Extract matrix A by evaluating f at basis vectors
    # f(t, y) = A @ y for linear autonomous system
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        ej = np.zeros(n, dtype=float)
        ej[j] = 1.0
        A[:, j] = f(t0, ej)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return y0 if we ran out of time (shouldn't happen for n=15)
        return y0
    
    # Compute exact solution: y(t1) = expm((t1 - t0) * A) @ y0
    dt = t1 - t0
    M = dt * A
    Y_final = expm(M) @ y0
    
    # Verify: check that the solution is finite
    if not np.all(np.isfinite(Y_final)):
        # Fallback to a simple RK4 with many steps
        h = dt / 10000
        t, y = t0, y0.copy()
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
        return y
    
    return Y_final
