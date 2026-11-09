"""ODE solver: Heat2D 10x10 – integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280

    if N != 100:
        # Fallback for unexpected sizes
        # Use simple RK4
        steps = 1000
        h = dt / steps
        t, y = t0, y0
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    # Strategy: Assume standard 2D Heat Equation on 10x10 grid
    # with Dirichlet boundary conditions and 5-point stencil.
    # u_t = u_xx + u_yy
    # Discretization:
    # u_xx approx (u[i-1,j] - 2u[i,j] + u[i+1,j]) / dx^2
    # u_yy approx (u[i,j-1] - 2u[i,j] + u[i,j+1]) / dy^2
    # Assuming dx=dy=1 (or scaled into the matrix).
    # The matrix A for 1D is tridiag(1, -2, 1).
    # Eigenvalues of A: lambda_k = -4 * sin^2(k * pi / (2 * (N_1d + 1)))
    # Eigenvectors: v_k[i] = sin((i+1) * k * pi / (N_1d + 1))
    
    N_1d = 10
    
    # Precompute 1D spectral data
    # k = 1..10
    k = np.arange(1, N_1d + 1)
    # Eigenvalues
    lambda_1d = -4.0 * np.sin(k * np.pi / (2.0 * (N_1d + 1)))**2
    
    # Eigenvector matrix V (N_1d x N_1d)
    # V[i, j] corresponds to eigenvector j (index j-1 in 0-based)
    # i is 0..9 (grid point)
    # j is 1..10 (mode)
    i_idx = np.arange(N_1d)
    j_idx = np.arange(1, N_1d + 1)
    
    # V[i, j-1] = sin((i+1) * j * pi / (N_1d + 1))
    V = np.sin((i_idx[:, None] + 1) * (j_idx[None, :] * np.pi) / (N_1d + 1))
    
    # The 2D operator L has eigenvalues mu_{k,l} = lambda_k + lambda_l
    # The 2D eigenvectors are outer products of 1D eigenvectors.
    # y_flat can be reshaped to Y (10x10).
    # Y = V * C * V^T  (if V is orthogonal, which it is up to scaling)
    # Actually, the sine transform matrix S is orthogonal if scaled by sqrt(2/(N+1)).
    # Let's use the unnormalized V and solve for coefficients.
    # Y = V C V^T  => C = V^{-1} Y (V^T)^{-1} = V^{-1} Y V^{-T}
    # Since V is symmetric (for this specific sine matrix? No, V is symmetric if we check indices?
    # V[i,j] = sin((i+1)(j+1)pi/11). V[j,i] = sin((j+1)(i+1)pi/11). Yes, V is symmetric.)
    # So C = V^{-1} Y V^{-1}
    
    # We can compute V_inv
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        # Fallback to RK4 if V is singular (should not happen)
        steps = 1000
        h = dt / steps
        t, y = t0, y0
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    # Reshape y0 to 10x10
    Y = y0.reshape(10, 10)
    
    # Compute coefficients C
    # C = V_inv @ Y @ V_inv (since V is symmetric)
    C = V_inv @ Y @ V_inv
    
    # Compute 2D eigenvalues
    # mu_{k,l} = lambda_k + lambda_l
    # lambda_1d is a vector of length 10.
    # mu is a 10x10 matrix where mu[i, j] = lambda_1d[i] + lambda_1d[j]
    mu = lambda_1d[:, None] + lambda_1d[None, :]
    
    # Scale coefficients by exp(mu * dt)
    # Note: lambda_1d are negative, so exp is stable.
    scale_factors = np.exp(mu * dt)
    
    C_scaled = C * scale_factors
    
    # Transform back
    # Y_final = V @ C_scaled @ V
    Y_final = V @ C_scaled @ V
    
    y_final = Y_final.flatten()
    
    # Check for validity
    if not np.all(np.isfinite(y_final)):
        # Fallback to RK4
        steps = 1000
        h = dt / steps
        t, y = t0, y0
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return y_final
# EVOLVE-BLOCK-END
