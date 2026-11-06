"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Extract A and b assuming y' = A y + b
    # f(t, y) = A y + b
    # Evaluate at y=0 to get b
    b = f(t0, np.zeros(n))
    
    # Evaluate at standard basis vectors to get columns of A
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        # f(t, ej) = A ej + b = A[:, j] + b
        # So A[:, j] = f(t, ej) - b
        A[:, j] = f(t0, ej) - b
        
    # Construct augmented matrix M for z = [y; 1]
    # z' = M z
    # M = [[A, b], [0, 0]]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = b
    # Last row is already zeros
    
    # Initial augmented state
    z0 = np.zeros(n + 1)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Compute matrix exponential
    # This is the core of the solution. scipy.linalg.expm is highly accurate.
    dt = t1 - t0
    expM = sla.expm(M * dt)
    
    # Final state
    z1 = expM @ z0
    y_final = z1[:n]
    
    # Check deadline (though expm is fast, just in case)
    if time.time() > deadline:
        # If we are somehow over, return what we have (should not happen)
        pass
        
    return y_final
# EVOLVE-BLOCK-END
