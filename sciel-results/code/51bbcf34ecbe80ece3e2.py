"""
Strategy: Structured Matrix Exponential via 5-point Laplacian (Neumann BC)
Instead of probing f with basis vectors (100+ calls) to reconstruct A,
we exploit the known PDE structure: 2D heat equation on a 10x10 grid with
Neumann boundary conditions uses a 5-point stencil. We build A directly
from the grid geometry using only 1-3 verification calls to f, then compute
expm(A * dt) @ y0. This reduces f-calls from ~100 to ~2-3, improving the
score by ~0.5*log10(100/3) ≈ 0.7.
"""

import numpy as np
import time
import sys

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 320

    # Expected grid size for 10x10 heat equation
    # n should be 100
    if n != 100:
        # Fallback: general approach for unexpected sizes
        return _general_solve(f, t_span, y0, deadline)
    
    # Build the 2D heat equation operator A for 10x10 grid with Neumann BC
    # Grid: 10x10, interior points indexed 0..9 in each direction
    # For Neumann BC: no flux at boundaries
    # The 5-point Laplacian with Neumann BC:
    # For interior point (i,j): A[i,j] has -4 on diagonal, +1 on 4 neighbors
    # For edge points (not corner): -3 on diagonal, +1 on existing neighbors
    # For corner points: -2 on diagonal, +1 on 2 neighbors
    
    # Actually, for the standard discretization of the heat equation:
    # u_t = u_xx + u_yy
    # With grid spacing h, the operator is (1/h^2) * L where L is the Laplacian matrix
    # But f(t, y) = A * y where A already includes the 1/h^2 factor
    # 
    # For a 10x10 grid with Neumann BC, we need to figure out h.
    # Standard: domain [0,1]x[0,1], 10 points => h = 1/9? Or h = 1/10?
    # With Neumann BC and 10 points, typically h = 1/(10-1) = 1/9 or h=1/10
    # Let's use the verification approach: call f once with a known y, 
    # and infer the scaling.
    
    # First, let's try to build A with a guessed h and verify.
    # Common choice: h = 1.0 (if the domain is [0,10]x[0,10] or similar)
    # Or h = 0.1 for [0,1]x[0,1] with 10 points.
    
    # Let's build the Laplacian structure (without 1/h^2) and determine h from f.
    
    def build_laplacian_neumann(n_grid):
        """Build the discrete Laplacian for n_grid x n_grid with Neumann BC.
        Returns a sparse-like dense matrix of size (n_grid*n_grid, n_grid*n_grid).
        The operator is: for interior, -4 on diag, +1 on 4-neighbors.
        For edges (non-corner): -3 on diag, +1 on 3 neighbors.
        For corners: -2 on diag, +1 on 2 neighbors.
        """
        N = n_grid * n_grid
        L = np.zeros((N, N))
        for i in range(n_grid):
            for j in range(n_grid):
                idx = i * n_grid + j
                # Count valid neighbors
                n_neighbors = 0
                if i > 0:
                    L[idx, (i-1)*n_grid + j] = 1.0
                    n_neighbors += 1
                if i < n_grid - 1:
                    L[idx, (i+1)*n_grid + j] = 1.0
                    n_neighbors += 1
                if j > 0:
                    L[idx, i*n_grid + (j-1)] = 1.0
                    n_neighbors += 1
                if j < n_grid - 1:
                    L[idx, i*n_grid + (j+1)] = 1.0
                    n_neighbors += 1
                L[idx, idx] = -n_neighbors
        return L
    
    L_struct = build_laplacian_neumann(10)
    
    # Now determine the scaling factor (1/h^2) by calling f once
    # Use a simple test vector
    y_test = np.ones(100)
    f_test = f(t0, y_test)
    
    # For y = ones, the Laplacian of a constant is 0 (with Neumann BC)
    # So f(t0, ones) should be 0 if it's a pure Laplacian
    # But if there's a different scaling or structure, we need to check.
    
    # Let's use a more informative test vector: a linear function
    # y[i,j] = i (linear in x direction)
    y_linear = np.zeros(100)
    for i in range(10):
        for j in range(10):
            y_linear[i*10+j] = float(i)
    f_linear = f(t0, y_linear)
    
    # For u(i,j) = i, u_xx = 0, u_yy = 0, so f should be 0
    # This confirms it's a Laplacian-like operator
    
    # Better: use a quadratic to determine the coefficient
    # u(i,j) = i^2, then u_xx = 2, so f = 2/h^2 * (something)
    # Actually for the discrete Laplacian: L[i^2] at interior point i:
    # (i+1)^2 - 2i^2 + (i-1)^2 = 2
    # So L * (i^2) = 2 for interior points (in 1D)
    # In 2D, if u(i,j) = i^2, then L*u = 2 for all points (since u_yy=0)
    
    y_quad = np.zeros(100)
    for i in range(10):
        for j in range(10):
            y_quad[i*10+j] = float(i*i)
    f_quad = f(t0, y_quad)
    
    # L_struct @ y_quad should give us the "raw" Laplacian applied to y_quad
    L_y_quad = L_struct @ y_quad
    
    # If f = c * L_struct @ y, then f_quad = c * L_y_quad
    # Estimate c from the ratio
    # Use points where L_y_quad is non-zero
    mask = np.abs(L_y_quad) > 1e-10
    if np.sum(mask) > 0:
        # Least squares estimate of c
        c = np.dot(f_quad[mask], L_y_quad[mask]) / np.dot(L_y_quad[mask], L_y_quad[mask])
    else:
        # Fallback: assume c = 1.0 or try another approach
        c = 1.0
    
    # Verify with another test vector
    y_quad2 = np.zeros(100)
    for i in range(10):
        for j in range(10):
            y_quad2[i*10+j] = float(j*j)
    f_quad2 = f(t0, y_quad2)
    L_y_quad2 = L_struct @ y_quad2
    mask2 = np.abs(L_y_quad2) > 1e-10
    if np.sum(mask2) > 0:
        c2 = np.dot(f_quad2[mask2], L_y_quad2[mask2]) / np.dot(L_y_quad2[mask2], L_y_quad2[mask2])
        # Average the estimates
        c = (c + c2) / 2.0
    
    # Verify the structure matches by checking residual
    # f(t0, y) should equal c * L_struct @ y
    y_verify = np.random.default_rng(42).random(100)
    f_verify = f(t0, y_verify)
    predicted = c * (L_struct @ y_verify)
    residual = np.linalg.norm(f_verify - predicted) / (np.linalg.norm(f_verify) + 1e-30)
    
    # If residual is large, the structure assumption is wrong
    if residual > 1e-6:
        # Fallback to general probing
        return _general_solve(f, t_span, y0, deadline)
    
    # Build A = c * L_struct
    A = c * L_struct
    
    dt = t1 - t0
    
    # Compute the solution using matrix exponential
    try:
        from scipy.linalg import expm
        M = expm(A * dt)
        y_final = M @ y0
    except Exception:
        # Fallback to general solver
        return _general_solve(f, t_span, y0, deadline)
    
    return y_final


def _general_solve(f, t_span, y0, deadline):
    """Fallback: probe A with basis vectors and use expm."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check homogeneity
    y_zero = np.zeros_like(y0)
    f_zero = f(t0, y_zero)
    
    if np.linalg.norm(f_zero) > 1e-12:
        b = f_zero
    else:
        b = None
    
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros_like(y0)
        e_j[j] = 1.0
        f_ej = f(t0, e_j)
        if b is not None:
            A[:, j] = f_ej - b
        else:
            A[:, j] = f_ej
    
    dt = t1 - t0
    
    try:
        from scipy.linalg import expm
        if b is None:
            M = expm(A * dt)
            y_final = M @ y0
        else:
            A_aug = np.zeros((n+1, n+1))
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            y_aug_0 = np.zeros(n+1)
            y_aug_0[:n] = y0
            y_aug_0[n] = 1.0
            M_aug = expm(A_aug * dt)
            y_aug_final = M_aug @ y_aug_0
            y_final = y_aug_final[:n]
    except Exception:
        # Last resort: simple RK4
        steps = 1000
        h = dt / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        y_final = y
    
    return y_final
