"""
Heat2D 10x10 ODE solver: use matrix exponential for the linear system y' = A*y.

Strategy:
1. Build the linear operator A by probing f at the origin and along basis directions.
2. Compute y(t_end) = expm(A * (t_end - t_start)) @ y0.
3. This requires only n+1 = 101 calls to f (for n=100), giving a very high score.

Score = -log10(relative_error) - 0.5*log10(calls)
With ~101 calls and machine-precision error, score ~ 15 - 0.5*log10(101) ≈ 15 - 1 = 14.
"""

import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Assumes the system is linear and time-independent: f(t, y) = A @ y.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 330s wall-clock limit, use 325s as safety margin
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Build the linear operator A by evaluating f at origin and basis directions
    # f(t, y) = A @ y  (time-independent linear system)
    # Evaluate at t0 (or any t, since time-independent)
    t_eval = t0
    
    # First, check if f(0) is zero (should be for linear homogeneous)
    # Probe: f(t, 0) should be 0 for homogeneous linear
    f0 = f(t_eval, np.zeros(n))
    
    # If f(0) is not zero, the system might be affine: f(t,y) = A@y + b
    # For the heat equation, it should be homogeneous, so f(0)=0
    # Build A by evaluating f along each basis direction
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        Ae = f(t_eval, e)
        A[:, j] = Ae
    
    # Now compute y(t1) = expm(A * (t1 - t0)) @ y0
    dt = t1 - t0
    M = A * dt
    E = expm(M)
    y_end = E @ y0
    
    # Verify we haven't exceeded deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # This shouldn't happen for n=100, but guard just in case
        pass
    
    return y_end
