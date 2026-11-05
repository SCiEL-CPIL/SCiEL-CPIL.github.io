"""ODE solver: 15-dimensional stiff linear system using Cayley-Hamilton reduction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) for a 15-dimensional stiff linear system on [t0, t1].
    
    Strategy: 
    1. Probe f to determine the constant matrix A (time-invariant linear system).
    2. Compute the matrix exponential exp(A * dt) using Cayley-Hamilton theorem
       to reduce to a polynomial of degree 14.
    3. Apply the matrix exponential to y0.
    
    This requires only 15 calls to f to determine A, plus 1 for the initial state.
    Total calls: ~16-18, which is minimal.
    """
    t0, t1 = t_span
    dt = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # If n is not 15, fall back to a simple method
    if n != 15:
        # Fallback: use scipy's solve_ivp if available, else RK4
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
            return sol.y[:, -1]
        except Exception:
            # Simple RK4 fallback
            steps = 10000
            h = dt / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
    
    # Step 1: Determine the constant matrix A by probing f.
    # For a linear system y' = A*y (time-invariant), f(t, y) = A @ y.
    # We can determine A by evaluating f at basis vectors.
    
    # First, get f(t0, y0) to check if it's consistent with linearity
    fy0 = f(t0, y0)
    
    # Probe with standard basis vectors to determine A
    # A @ e_j = f(t, e_j) for any t (since time-invariant)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Verify linearity: check that f(t0, y0) ≈ A @ y0
    # If not linear or not time-invariant, we need a different approach
    residual = np.linalg.norm(fy0 - A @ y0)
    rel_residual = residual / (np.linalg.norm(A @ y0) + 1e-300)
    
    if rel_residual > 1e-8:
        # System might not be purely linear or might have time-dependence
        # Try to check time-invariance by probing at different times
        # If time-varying, fall back to implicit method
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
            return sol.y[:, -1]
        except Exception:
            # Backward Euler fallback
            steps = 5000
            h = dt / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                # Implicit Euler with Newton
                z = y.copy()
                for _newton in range(10):
                    fz = f(t + h, z)
                    # Finite difference Jacobian
                    J = np.empty((n, n))
                    for jj in range(n):
                        e = np.zeros(n)
                        e[jj] = 1e-8 * max(1.0, abs(z[jj]))
                        J[:, jj] = (f(t + h, z + e) - fz) / e[jj]
                    try:
                        step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
                    except np.linalg.LinAlgError:
                        break
                    z = z + step
                    if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z)):
                        break
                y, t = z, t + h
            return y
    
    # Step 2: Compute exp(A * dt) using Cayley-Hamilton theorem
    # The matrix exponential can be expressed as a polynomial of degree n-1:
    # exp(A*dt) = c0*I + c1*A + c2*A^2 + ... + c_{n-1}*A^{n-1}
    # where coefficients are determined by matching the eigenvalues.
    
    # Find eigenvalues of A
    eigenvalues = np.linalg.eigvals(A)
    
    # Build Vandermonde-like system to solve for coefficients
    # For each eigenvalue lambda_i: exp(lambda_i * dt) = sum_{k=0}^{n-1} c_k * lambda_i^k
    # This is a Vandermonde system: V @ c = exp(eigenvalues * dt)
    
    # Handle potential numerical issues with repeated eigenvalues
    # Use SVD-based least squares for robustness
    V = np.vander(eigenvalues, N=n, increasing=True)  # V[i, k] = eigenvalues[i]^k
    rhs = np.exp(eigenvalues * dt)
    
    # Solve for coefficients
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(V, rhs, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to direct solve
        coeffs = np.linalg.solve(V, rhs)
    
    # Step 3: Compute the matrix polynomial
    # exp(A*dt) = sum_{k=0}^{n-1} coeffs[k] * A^k
    A_power = np.eye(n)
    exp_A_dt = np.zeros((n, n))
    for k in range(n):
        exp_A_dt += coeffs[k] * A_power
        A_power = A_power @ A
    
    # Step 4: Apply to initial condition
    y_final = exp_A_dt @ y0
    
    return y_final
# EVOLVE-BLOCK-END
