"""ODE solver for convection-diffusion 100-point problem exploiting tridiagonal structure."""
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
    
    # 1. Detect linearity and extract A, b using tridiagonal structure assumption.
    # For a 1D convection-diffusion problem with 100 points, the operator is tridiagonal.
    # y' = A y + b
    # We need to find A and b.
    # Call f(t, 0) to get b.
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    
    # To extract A, we need to call f at specific points.
    # Since A is tridiagonal, we only need to determine the 3 diagonals.
    # Let A have diagonals d_main, d_sub, d_super.
    # Row i of A: A[i, i-1] = d_sub, A[i, i] = d_main, A[i, i+1] = d_super (with boundary adjustments).
    # Actually, let's just extract the first few columns and rows to determine the pattern,
    # or better yet, just call f at e_0, e_1, e_{n-2}, e_{n-1} to get the corners and assume uniformity?
    # No, the coefficients might vary if the grid is non-uniform or coefficients are variable.
    # However, standard 1D convection-diffusion on a uniform grid has constant coefficients.
    # Let's assume constant coefficients for the tridiagonal matrix.
    # A = tridiag(c, d, u)
    
    # We can determine c, d, u by calling f at specific basis vectors.
    # f(t, e_0) = A[:, 0] + b = [d, c, 0, ...] + b  (assuming e_0 is index 0)
    # Wait, A @ e_j gives the j-th column of A.
    # Column 0: [d, c, 0, ..., 0]^T + b
    # Column 1: [u, d, c, 0, ..., 0]^T + b
    # Column n-1: [0, ..., 0, c, d]^T + b
    
    # Let's call f at e_0, e_1, e_{n-1}.
    # This is only 3 calls + 1 for b = 4 calls.
    
    e0 = np.zeros(n); e0[0] = 1.0
    e1 = np.zeros(n); e1[1] = 1.0
    en1 = np.zeros(n); en1[n-1] = 1.0
    
    f_e0 = f(t0, e0)
    f_e1 = f(t0, e1)
    f_en1 = f(t0, en1)
    
    # A[:, 0] = f_e0 - b
    # A[:, 1] = f_e1 - b
    # A[:, n-1] = f_en1 - b
    
    col0 = f_e0 - b
    col1 = f_e1 - b
    coln1 = f_en1 - b
    
    # From col0:
    # A[0,0] = col0[0]
    # A[1,0] = col0[1]
    # A[i,0] = 0 for i > 1
    d = col0[0]
    c = col0[1]  # sub-diagonal
    
    # From col1:
    # A[0,1] = col1[0]
    # A[1,1] = col1[1]
    # A[2,1] = col1[2]
    # A[i,1] = 0 for i > 2
    u = col1[0]  # super-diagonal
    d_check = col1[1]
    c_check = col1[2]
    
    # From coln1:
    # A[n-2, n-1] = coln1[n-2]
    # A[n-1, n-1] = coln1[n-1]
    # A[i, n-1] = 0 for i < n-2
    c_check2 = coln1[n-2]
    d_check2 = coln1[n-1]
    
    # Verify consistency
    if not (np.isclose(d, d_check) and np.isclose(d, d_check2) and 
            np.isclose(c, c_check) and np.isclose(c, c_check2) and 
            np.isclose(u, coln1[n-2] if n > 2 else 0)): # Note: coln1[n-2] is A[n-2, n-1] which is c.
        # If not consistent, fall back to full extraction or RK4
        # For now, let's just proceed with the first estimates if they are close, or use RK4.
        # Given the task is "convection-diffusion 100-point", it's almost certainly uniform.
        pass
    
    # Construct A
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = d
        if i > 0:
            A[i, i-1] = c
        if i < n-1:
            A[i, i+1] = u
            
    # Verify linearity with a random vector (1 more call)
    # Actually, we can skip this if we are confident, but it's safer.
    # Let's do a quick check with a random vector.
    y_rand = np.random.randn(n)
    f_pred = A @ y_rand + b
    f_actual = f(t0, y_rand)
    
    if not np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
        # Fallback: extract full A (100 calls) or RK4
        # Let's extract full A to be safe, but that costs 100 calls.
        # Or just use RK4. RK4 is robust.
        # But wait, if the structure is NOT tridiagonal, RK4 is the safe bet.
        # However, the problem statement says "convection-diffusion 100-point", which implies the standard discretization.
        # Let's stick with the tridiagonal assumption if the check fails, it might be due to floating point or non-uniform grid.
        # If non-uniform, we need to extract more.
        # For now, let's assume the check passes. If it doesn't, we use RK4.
        return _rk4_fallback(f, t_span, y0)

    # 2. Solve y' = A y + b using matrix exponential
    dt = t1 - t0
    E = expm(A * dt)
    
    # Compute integral term: \int_0^dt expm(A*u) b du
    # z(dt) = sum_{k=0}^inf (A^k b dt^{k+1}) / (k+1)!
    z = np.zeros(n)
    term = b.copy() * dt
    for k in range(60):
        z += term
        if np.linalg.norm(term) < 1e-15 * (1 + np.linalg.norm(z)):
            break
        term = A @ term * (dt / (k + 2))
        
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
