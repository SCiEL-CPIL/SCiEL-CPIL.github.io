"""
ODE solver: 15-dimensional stiff linear system.

Strategy:
- Call f once at (0, y0) to get the initial derivative.
- Use a Taylor series expansion of the state:
    y(t) = sum_{k=0}^{K} (t^k / k!) * y^{(k)}(0)
  where y^{(k+1)}(0) = A^T * y^{(k)}(0) for a linear system y' = A y + b.
- For a linear system, we can extract A and b from a few evaluations,
  but to keep f-call count minimal we instead use the fact that for
  a linear autonomous system y' = A*y + b, the derivatives at t=0 can
  be generated recursively IF we can compute A*y + b. Since we only
  have f, we can use the following trick:
  
  For a linear system, f(t, y) = A*y + b (assuming time-independent).
  We can estimate A and b by calling f at y0 and a perturbed y.
  But the directive says to use minimal 16 calls to identify A and b.
  
  Actually, a simpler and more robust approach for a 15-dim system:
  Use the fact that if the system is linear and time-independent, we can
  use the Krylov subspace or simply compute the Taylor coefficients by
  repeated application of the linear operator. Since we can't get A
  directly without many calls, let's use a different strategy:
  
  Use a high-order Runge-Kutta or Taylor method with automatic step
  size control, leveraging the fact that for stiff linear systems,
  implicit methods or matrix exponential methods are ideal.
  
  The directive specifically mentions Taylor series of the matrix
  exponential. So let's extract A and b first using 16 calls (1 for
  f(0,y0), and 15 more for column vectors of A), then compute
  y(1) = expm(A)*y0 + A^{-1}*(expm(A) - I)*b.
"""

import numpy as np


def _matrix_exp_taylor(M, tol=1e-15):
    """
    Compute matrix exponential using scaling and squaring with Taylor series.
    """
    n = M.shape[0]
    norm_M = np.linalg.norm(M, ord=1)
    if norm_M == 0:
        return np.eye(n)
    
    # Choose scaling factor s such that norm(M / 2^s) < 0.5
    s = max(0, int(np.ceil(np.log2(norm_M) + 1)))
    
    # Scale the matrix
    M_scaled = M / (2 ** s)
    
    # Taylor series
    term = np.eye(n)
    E = term.copy()
    
    for k in range(1, 100):
        term = (term @ M_scaled) / k
        E += term
        
        # Check convergence
        if np.linalg.norm(term, ord=1) < tol * np.linalg.norm(E, ord=1):
            break
            
    # Unscale by squaring s times
    for _ in range(s):
        E = E @ E
        
    return E


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1]
    with initial condition y0.
    
    Assumes a 15-dimensional linear time-invariant system.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract the linear operator A and offset b from f.
    y0_orig = y0.copy()
    f0 = np.array(f(t0, y0), dtype=float)
    
    # Estimate A using forward differences
    eps = 1e-8
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        y_pert = y0_orig.copy()
        y_pert[j] += eps
        f_pert = np.array(f(t0, y_pert), dtype=float)
        A[:, j] = (f_pert - f0) / eps
    
    # b = f0 - A @ y0
    b = f0 - A @ y0_orig
    
    h = t1 - t0
    
    # Construct the augmented matrix for the system:
    # d/dt [y; 1] = [A  b; 0 0] [y; 1]
    M_aug = np.zeros((n + 1, n + 1), dtype=float)
    M_aug[:n, :n] = A
    M_aug[:n, n] = b
    
    # Compute exponential of h * M_aug
    M_to_exp = h * M_aug
    
    E_aug = _matrix_exp_taylor(M_to_exp)
    
    # Initial vector
    v0 = np.zeros(n + 1)
    v0[:n] = y0_orig
    v0[n] = 1.0
    
    v_final = E_aug @ v0
    
    return v_final[:n]
