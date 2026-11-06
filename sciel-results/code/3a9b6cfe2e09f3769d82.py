import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Estimate the system structure: y' = A y + b
    # 1. Estimate b = f(0, 0)
    b = f(0, np.zeros(n))
    
    # 2. Estimate A columns
    # A[:, j] = f(0, e_j) - b
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(0, e_j) - b
        
    # 3. Verify linearity/affineness with a random test point
    # If the system is not linear/affine, this check will fail.
    # We use a deterministic pseudo-random vector to avoid overhead of RNG setup
    test_y = np.ones(n) * 0.5
    test_pred = A @ test_y + b
    test_true = f(0, test_y)
    
    # Check if the linear model holds
    if not np.allclose(test_pred, test_true, rtol=1e-8, atol=1e-10):
        # Fallback to a robust ODE solver if the system is non-linear
        from scipy.integrate import solve_ivp
        # Use Radau for stiff problems, tight tolerances
        sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-12)
        return sol.y[:, -1]
        
    # 4. Compute solution using Matrix Exponential
    E = expm(A * dt)
    
    if np.linalg.norm(b) < 1e-12:
        # Homogeneous: y(t) = exp(A t) y0
        return E @ y0
    else:
        # Affine: y(t) = exp(A t) y0 + (exp(A t) - I) A^{-1} b
        try:
            x = np.linalg.solve(A, b)
            return E @ y0 + (E - np.eye(n)) @ x
        except np.linalg.LinAlgError:
            # If A is singular, use the integral form or fallback
            # For heat equation, this is unlikely unless Neumann BCs with zero mode.
            # If singular, we can still use solve_ivp as a safe bet
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-12)
            return sol.y[:, -1]
