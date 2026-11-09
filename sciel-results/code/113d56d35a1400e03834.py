"""ODE solver for convection-diffusion 100-point problem using matrix exponential strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Safety deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    # 1. Detect linearity and time-invariance, and extract A, b
    # For y' = f(t, y), if linear and time-invariant: f(t, y) = A @ y + b
    # We can determine A and b by calling f at specific points.
    
    y0_test = np.zeros(n)
    b = f(t0, y0_test)
    
    # Extract A columns
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        # f(t, ej) = A @ ej + b = A[:, j] + b
        # So A[:, j] = f(t, ej) - b
        A[:, j] = f(t0, ej) - b
        
    # Verify linearity with a random vector
    y_rand = np.random.randn(n)
    y_rand[0] = 1.0  # Ensure non-zero
    f_pred = A @ y_rand + b
    f_actual = f(t0, y_rand)
    
    if not np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
        # Fallback to RK4 if not linear
        return _rk4_fallback(f, t_span, y0)
    
    # 2. Solve y' = A y + b using augmented matrix exponential
    # The solution is y(t) = expm(A*dt) @ y0 + z where z' = A*z + b, z(0) = 0
    # We compute z using an augmented system:
    # Let w = [z; 1], then w' = [[A, b], [0, 0]] @ w
    # So w(dt) = expm([[A, b], [0, 0]] * dt) @ [0; 1]
    # And z(dt) = w(dt)[:n]
    
    dt = t1 - t0
    
    # Build augmented matrix of size (n+1) x (n+1)
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    # Last row is all zeros (for the constant 1)
    
    # Initial condition for augmented system: [0; 1]
    w0 = np.zeros(n + 1)
    w0[n] = 1.0
    
    # Compute matrix exponential of augmented system
    E_aug = expm(A_aug * dt)
    
    # Final augmented state
    w_final = E_aug @ w0
    
    # Extract z and combine with homogeneous solution
    z = w_final[:n]
    E = E_aug[:n, :n]  # This is expm(A * dt)
    
    y_final = E @ y + z
    
    # Check time limit
    if time.time() > deadline:
        return y_final
    
    return y_final

def _rk4_fallback(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 1000
    t = t0
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
