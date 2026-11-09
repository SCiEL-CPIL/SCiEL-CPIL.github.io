"""ODE solver for convection-diffusion 100-point problem using exact linear algebra strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm, solve
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
    # We determine A and b by calling f at specific points.
    
    # Call f at t0 with y=0 to get b (assuming linearity)
    y0_test = np.zeros(n)
    b = np.array(f(t0, y0_test), dtype=float)
    
    # Extract A columns
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        # f(t, ej) = A @ ej + b = A[:, j] + b
        # So A[:, j] = f(t, ej) - b
        A[:, j] = np.array(f(t0, ej), dtype=float) - b
        
    # Verify linearity with a random vector
    y_rand = np.random.randn(n)
    y_rand[0] = 1.0  # Ensure non-zero
    f_pred = A @ y_rand + b
    f_actual = np.array(f(t0, y_rand), dtype=float)
    
    if not np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
        # Fallback to RK4 if not linear
        return _rk4_fallback(f, t_span, y0)
    
    # Check time limit
    if time.time() > deadline:
        # If we are out of time, we can't compute expm. 
        # Return a rough estimate? Or just y0? 
        # Actually, expm is usually fast for n=100. Let's try to be safe.
        # If we haven't computed E yet, we can't return a good answer.
        # But this check is before expm.
        pass

    # 2. Solve y' = A y + b using exact linear algebra
    # Solution: y(t) = e^{A(t-t0)} y(t0) + \int_{t0}^t e^{A(t-s)} b ds
    # Let T = t1 - t0.
    # y(t1) = e^{AT} y0 + \int_0^T e^{A(T-u)} b du  (change of variable u = T - (t-s))
    # Wait, the standard formula is:
    # y(t) = e^{A(t-t0)} y0 + \int_{t0}^t e^{A(t-s)} b ds
    # Let u = t-s, du = -ds. Limits: s=t0 -> u=T, s=t -> u=0.
    # \int_T^0 e^{Au} b (-du) = \int_0^T e^{Au} b du
    
    # We need to compute E = expm(A * T) and Z = \int_0^T expm(A * u) b du
    
    T = t1 - t0
    
    # Compute E = expm(A * T)
    # This is the most expensive part, but n=100 is small.
    E = expm(A * T)
    
    # Compute Z = \int_0^T expm(A * u) b du
    # Note that Z is the solution to w' = A w + b, w(0)=0, evaluated at T.
    # Also, note that Z = A^{-1} (E - I) b if A is invertible.
    # If A is singular, we can use the augmented system or series.
    # However, for convection-diffusion, A is typically singular (e.g., Neumann BCs) or ill-conditioned.
    # A more robust way is to solve the linear system:
    # Consider the augmented matrix M = [[A, -I], [I, 0]]? No.
    
    # Let's use the property:
    # Z = \int_0^T e^{Au} b du
    # Differentiate: Z' = A Z + b, Z(0)=0.
    # We can compute this by solving the Sylvester-like system or using the identity:
    # Z = (E - I) A^{-1} b (if A invertible)
    
    # To handle singular A, we can use the following trick:
    # Solve [A, -I; I, 0] [Z; b] = [0; E-I]? No.
    
    # Alternative robust method:
    # Z is the last column of the solution to X' = A X + [0...0 b], X(0)=0?
    # Actually, we can just use the series expansion which is safe if we stop early, 
    # OR we can use the fact that for any matrix A, 
    # \int_0^T e^{Au} du = \sum_{k=0}^{\infty} \frac{A^k T^{k+1}}{(k+1)!}
    # This series converges for all A. The previous code had a bug in the recurrence.
    # Let's implement the series correctly.
    
    # T_k = A^k b T^{k+1} / (k+1)!
    # T_0 = b * T
    # T_{k+1} = A @ T_k * T / (k+2)
    
    Z = np.zeros(n)
    term = b * T
    for k in range(100): # 100 terms is plenty for n=100
        Z += term
        # Check convergence
        norm_term = np.linalg.norm(term)
        norm_Z = np.linalg.norm(Z)
        if norm_term < 1e-15 * (1.0 + norm_Z):
            break
        # Update term for next iteration
        # term_{k+1} = A @ term_k * T / (k+2)
        term = (A @ term) * (T / (k + 2))
        
    y_final = E @ y + Z
    
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
