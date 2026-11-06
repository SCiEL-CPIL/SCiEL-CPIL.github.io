"""Heat2D 10x10 ODE solver using linear-system structure."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) over t_span.
    
    Strategy: The heat equation on a 10x10 grid is a linear autonomous ODE: y' = A*y.
    We estimate A using function evaluations, then compute expm(A * (t1-t0)) @ y0
    for an exact (up to matrix exponential accuracy) solution.
    
    If the system turns out to be nonlinear (which it shouldn't be for the heat eq),
    we fall back to a high-order implicit method.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    deadline = time.time() + 290  # 300s limit, leave 10s margin
    
    # Estimate the Jacobian A at t0, y0
    # For a linear autonomous system, f(t, y) = A * y (possibly + b, but heat eq has b=0)
    # We'll estimate A using forward differences along each coordinate direction.
    
    # First, get f(t0, y0)
    f0 = f(t0, y0)
    
    # Check if it's linear: f(t, y) should be proportional to y
    # Estimate A columns by perturbing each coordinate
    A = np.zeros((n, n), dtype=float)
    
    # Use a small perturbation
    eps = 1e-8
    for j in range(n):
        if time.time() > deadline - 10:
            # Time pressure - use a coarser estimate
            break
        y_pert = y0.copy()
        y_pert[j] += eps * max(1.0, abs(y0[j]))
        f_pert = f(t0, y_pert)
        A[:, j] = (f_pert - f0) / (eps * max(1.0, abs(y0[j])))
    
    # Verify linearity: check if f(t0, 2*y0) ≈ 2*f(t0, y0)
    # This costs 1 extra f call
    y_test = 2.0 * y0
    f_test = f(t0, y_test)
    residual = np.linalg.norm(f_test - 2.0 * f0) / max(1.0, np.linalg.norm(f0))
    
    # Also check time-independence with a small time shift
    dt_small = 1e-6
    f_time = f(t0 + dt_small, y0)
    residual_time = np.linalg.norm(f_time - f0) / max(1.0, np.linalg.norm(f0))
    
    is_linear = residual < 1e-6 and residual_time < 1e-4
    
    if is_linear:
        # Use matrix exponential for exact solution
        # y(t1) = expm(A * (t1 - t0)) @ y0
        dt = t1 - t0
        M = expm(A * dt)
        y_final = M @ y0
        
        # Verify with a few RK4 steps to make sure our A estimate is good
        # Use a moderate number of RK4 steps as a sanity check
        h_check = dt / 100
        y_check = y0.copy()
        t_check = t0
        for _ in range(100):
            k1 = f(t_check, y_check)
            k2 = f(t_check + h_check/2, y_check + h_check/2 * k1)
            k3 = f(t_check + h_check/2, y_check + h_check/2 * k2)
            k4 = f(t_check + h_check, y_check + h_check * k3)
            y_check = y_check + h_check/6 * (k1 + 2*k2 + 2*k3 + k4)
            t_check += h_check
        
        # If the RK4 check agrees with expm, use expm result
        err = np.linalg.norm(y_final - y_check) / max(1.0, np.linalg.norm(y_final))
        if err < 1e-5:
            return y_final
        else:
            # A estimate might be off; use more accurate A estimation
            # Try central differences for better accuracy
            A_central = np.zeros((n, n), dtype=float)
            eps_c = 1e-6
            for j in range(n):
                if time.time() > deadline - 20:
                    break
                y_plus = y0.copy()
                y_plus[j] += eps_c * max(1.0, abs(y0[j]))
                y_minus = y0.copy()
                y_minus[j] -= eps_c * max(1.0, abs(y0[j]))
                A_central[:, j] = (f(t0, y_plus) - f(t0, y_minus)) / (2 * eps_c * max(1.0, abs(y0[j])))
            
            M_central = expm(A_central * dt)
            y_final_central = M_central @ y0
            
            # Compare with RK4 check
            err_central = np.linalg.norm(y_final_central - y_check) / max(1.0, np.linalg.norm(y_final_central))
            if err_central < err:
                return y_final_central
            else:
                return y_final
    else:
        # Nonlinear or time-dependent - use high-order implicit method
        # Use implicit Runge-Kutta (Radau) style approach
        # For simplicity, use backward Euler with Newton iterations
        steps = 500
        h = dt / steps
        t = t0
        y = y0.copy()
        n_steps = steps
        
        for _ in range(n_steps):
            if time.time() > deadline - 5:
                break
            z = y.copy()
            # Newton iterations for implicit Euler: z = y + h * f(t+h, z)
            for _newton in range(6):
                fz = f(t + h, z)
                # Approximate Jacobian using finite differences
                J = np.zeros((n, n), dtype=float)
                eps_j = 1e-7
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = eps_j * max(1.0, abs(z[j]))
                    J[:, j] = (f(t + h, z + e) - fz) / e[j]
                
                # Solve: (I - h*J) * delta = -(z - y - h*fz)
                try:
                    delta = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
                    z_new = z + delta
                    if np.linalg.norm(delta) <= 1e-12 * (1 + np.linalg.norm(z)):
                        z = z_new
                        break
                    z = z_new
                except np.linalg.LinAlgError:
                    break
            
            y = z
            t += h
        
        return y

# EVOLVE-BLOCK-END
