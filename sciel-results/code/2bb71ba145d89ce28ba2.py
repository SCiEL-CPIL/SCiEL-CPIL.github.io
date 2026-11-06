"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Solve the fixed 10x10 2D heat equation IVP on [0, 0.05].
    
    The problem is a linear autonomous ODE: y' = A y, where A is the 2D Laplacian
    (discrete). We exploit the spectral decomposition of A via FFT to solve
    analytically in the frequency domain.
    
    Strategy:
    1. We need to identify the matrix A from f. Since f(t, y) = A @ y for the
       heat equation, we can determine A by evaluating f at a few basis vectors.
       However, the task says "at most 2000000 calls" and we want to minimize calls.
       
       Actually, looking more carefully: the problem is a FIXED 10x10 heat equation.
       The Laplacian matrix for a 10x10 grid with Dirichlet boundary conditions
       (or whatever standard discretization) is known. But we shouldn't assume 
       the exact discretization without checking.
       
       Let's think about this differently. The heat equation on a 10x10 grid with
       standard 5-point stencil gives a specific matrix. But the safest approach 
       is to determine A from f evaluations.
       
       Since f(t, y) = A @ y (linear autonomous), we can get A by calling f 
       with standard basis vectors. That's 100 calls for a 100-dimensional system.
       Then we diagonalize A, apply exp(-lambda_i * t) to each component, and
       reconstruct.
       
       Actually, for the 2D heat equation with the standard discretization, the
       Laplacian is a Kronecker sum of 1D Laplacians, which is diagonalizable by
       FFT (if periodic) or DCT (if Dirichlet). Let me check which boundary
       conditions are likely.
       
       For a standard 10x10 heat equation problem, it's typically Dirichlet
       boundary conditions with zero boundaries, and the interior is 10x10.
       The 1D Laplacian with Dirichlet BCs is tridiagonal with -2 on diagonal
       and 1 on off-diagonals. This is diagonalizable by DCT.
       
       But to be safe and general, let me just determine A from f and use
       eigendecomposition. 100 calls to get A, then one eigendecomposition,
       then apply the exponential. Total calls: 100 (or maybe fewer if we can
       be clever).
       
       Actually, let me try a different approach: since the problem is fixed
       and we know it's a 2D heat equation, let me try to infer the structure.
       
       Let me just use the direct approach: evaluate f on basis vectors to get A,
       then use eigendecomposition to solve analytically.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Determine the matrix A from f(t, y) = A @ y
    # Evaluate f at each standard basis vector
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    
    # Eigendecomposition of A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Transform y0 to eigenbasis
    c = np.linalg.solve(eigenvectors, y0)
    
    # Apply exponential decay: y(t) = V @ diag(exp(lambda_i * (t1-t0))) @ c
    dt = t1 - t0
    exp_factors = np.exp(eigenvalues * dt)
    
    # Reconstruct
    y_final = eigenvectors @ (exp_factors * c)
    
    return y_final
# EVOLVE-BLOCK-END
