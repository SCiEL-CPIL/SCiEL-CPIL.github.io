"""
Heat2D 10x10 ODE solver.

Strategy:
- The heat equation discretised on a 10x10 grid gives a linear, autonomous,
  stiff ODE: y' = A y (with possibly boundary handling, but linear).
- We estimate the matrix A using a small number of function evaluations.
  Since the system is linear and time-independent, one call gives A*y0,
  and we can estimate A by perturbing y0 in coordinate directions.
  However, 100 perturbations * 100 dim = 10000 f-calls just for A, which
  is fine within 2 million budget.
  
  Actually, a better approach: since the ODE is linear and autonomous, we can
  use the fact that expm(A*h) * y0 gives the exact solution. We estimate A
  using central differences or forward differences.

  But wait - the evaluator counts calls to f. Let's think about efficiency.

  Alternative: Use an implicit method that's very accurate for stiff linear problems.
  Radau IIA order 5 would be excellent but complex.

  Simplest high-accuracy approach for linear autonomous stiff systems:
  1. Estimate A with n f-calls (forward differences, one per column)
  2. Compute expm(A * h) using scipy
  3. Step through with large h values

  For n=100, estimating A costs ~100 f-calls. Then we can take very large steps
  since expm is exact for the linear system.

  Let's estimate A carefully:
  A[:,j] = (f(t, y + e_j) - f(t, y)) / eps_j
  
  But we need a good eps. For heat equation with 10x10 grid, values are O(1).
  Use eps = 1e-7 or adaptive.

  Actually, even simpler: since it's linear, f(t, y) = A*y for all y.
  We can estimate A by calling f with basis vectors:
  A[:,j] = f(t, e_j)
  
  This requires exactly n = 100 f-calls! And it's exact (up to floating point)
  for a linear system.

  Then the solution is y(t) = expm(A*(t-t0)) @ y0.
  
  We just need to call expm once and multiply. Total f-calls: ~100.
  Accuracy: limited only by expm precision and the linearity assumption.
  
  The heat equation is definitely linear (with fixed boundary conditions handled
  in the matrix or as part of the linear operator), so this should work perfectly.
"""

import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Estimate the system matrix A by probing with standard basis vectors.
    # For a linear autonomous system y' = A y, f(t, e_j) = A e_j = A[:, j]
    # This requires exactly n function evaluations.
    
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Compute the matrix exponential to propagate the solution exactly.
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0.copy()
    
    M = expm(A * dt)
    y_end = M @ y0
    
    # Safety: ensure finite result
    if not np.all(np.isfinite(y_end)):
        # Fallback: use a simple implicit method
        # This should never happen for well-posed heat equation
        y = y0.copy()
        h = dt / 10000
        for _ in range(10000):
            # Simple backward Euler with one Newton iteration (crude)
            fy = f(t0, y)
            # For linear system: (I - h*A) y_new = y
            try:
                y = np.linalg.solve(np.eye(n) - h * A, y)
            except:
                y = y - h * fy
        y_end = y
    
    return y_end
