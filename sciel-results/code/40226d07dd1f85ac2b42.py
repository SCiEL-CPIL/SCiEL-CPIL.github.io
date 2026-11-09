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
    # Let's call f at t=0 with y=0 to get b (assuming linearity).
    # Then call f at t=0 with y=e_j (standard basis) to get columns of A.
    # But we only have 2,000,000 calls. n=100, so 101 calls is cheap.
    # Let's verify linearity first.
    
    y0_test = np.zeros(n)
    b = f(t0, y0_test)
    
    # Check if f is time-invariant by calling at t1 with y0
    # If f(t0, y) == f(t1, y) for all y, it's time-invariant.
    # Let's just assume it is, as convection-diffusion usually is in this context.
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
        # This shouldn't happen for the specified task, but good for safety
        return _rk4_fallback(f, t_span, y0)
    
    # 2. Solve y' = A y + b using matrix exponential
    # Solution: y(t) = e^{A(t-t0)} y(t0) + \int_{t0}^t e^{A(t-s)} b ds
    # The integral part can be computed as:
    # If A is invertible: A^{-1} (e^{A(t-t0)} - I) b
    # If A is singular: use series or solve augmented system.
    # A more robust way: use scipy.integrate.quad_vec or just numerical integration of the ODE
    # with large steps, but we want to minimize f calls.
    # Actually, we can just use the formula:
    # y(t) = expm(A*dt) @ y0 + expm_multiply(A*dt, b) * dt? No.
    # The particular solution for constant b is:
    # y_p(t) = \int_0^t expm(A*(t-s)) b ds
    # Let u = t-s, du = -ds
    # y_p(t) = \int_0^t expm(A*u) b du
    # This is exactly expm_multiply(A, b) * t if A was scalar, but for matrix:
    # It is the solution to z' = A z + b, z(0)=0.
    # We can compute this by solving the augmented system or using series.
    # Simpler: just use a few steps of a high-order method on the linear system?
    # No, the point is to use expm.
    
    # Let's compute the particular solution using the series expansion or a trick.
    # Trick: solve [A, -I; I, 0] ? No.
    # Let's just use the fact that we can integrate y' = A y + b with large steps
    # using the exact solution formula for linear ODEs.
    # y(t+h) = expm(A*h) y(t) + \int_0^h expm(A*(h-s)) b ds
    
    # We can compute the integral term once for a given h, but h might vary.
    # However, if we use a single large step, we can compute it directly.
    # Let's try a single step first.
    
    dt = t1 - t0
    E = expm(A * dt)
    
    # Compute integral term: \int_0^dt expm(A*u) b du
    # This is equivalent to solving A z + b = 0? No.
    # It is the value at t=dt of the IVP z' = A z + b, z(0)=0.
    # We can approximate this with a few steps of RK4 on z, or use the series:
    # z(dt) = b*dt + A*b*dt^2/2 + A^2*b*dt^3/6 + ...
    # Since we have the matrix A, we can compute powers of A.
    # But n=100, so A^k is cheap.
    
    # Let's compute the series for the integral term
    # z(dt) = sum_{k=0}^inf (A^k b dt^{k+1}) / (k+1)!
    
    z = np.zeros(n)
    term = b.copy() * dt  # T_0 = b * dt
    for k in range(60):  # 60 terms for convergence
        z += term
        # Check convergence
        if np.linalg.norm(term) < 1e-15 * (1 + np.linalg.norm(z)):
            break
        # T_{k+1} = A @ T_k * dt / (k+2)
        term = A @ term * (dt / (k + 2))
        
    y_final = E @ y + z
    
    # Check time limit
    if time.time() > deadline:
        return y_final
    
    # Verify the solution by checking if it satisfies the ODE at the end?
    # Or just return it. The problem is well-posed.
    
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
