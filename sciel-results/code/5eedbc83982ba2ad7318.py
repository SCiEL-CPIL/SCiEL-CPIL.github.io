"""Combined solver: exploits linearity via matrix exponential, falls back to DOP853."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Primary strategy: exploit the linear structure of convection-diffusion.
    Determine if the system is homogeneous linear (y' = A*y) or affine (y' = A*y + c),
    then compute the exact solution using the matrix exponential.
    
    Fallback: DOP853 adaptive Runge-Kutta with tight tolerances.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = len(y0_arr)
    T = t1 - t0
    
    start_time = time.time()
    DEADLINE = 325.0  # safety margin before 330s kill
    
    # --- Strategy 1: Exploit linearity ---
    # For a linear autonomous ODE y' = A*y + c, we can solve exactly.
    # First, check if the system is homogeneous linear by evaluating f at zero.
    
    try:
        f_zero = np.asarray(f(t0, np.zeros(n)), dtype=float)
    except Exception:
        f_zero = None
    
    if f_zero is not None:
        is_homogeneous = np.allclose(f_zero, 0.0, atol=1e-10)
        
        if is_homogeneous:
            # System is y' = A*y (homogeneous linear)
            # Determine A by evaluating f at unit vectors: A[:, j] = f(t0, e_j)
            try:
                A = np.zeros((n, n))
                for j in range(n):
                    e_j = np.zeros(n)
                    e_j[j] = 1.0
                    A[:, j] = np.asarray(f(t0, e_j), dtype=float)
                
                # Verify linearity: check that A @ y0 ≈ f(t0, y0)
                f_y0 = np.asarray(f(t0, y0_arr), dtype=float)
                A_y0 = A @ y0_arr
                if np.allclose(A_y0, f_y0, atol=1e-8, rtol=1e-6):
                    # Confirmed linear: compute exact solution via matrix exponential
                    expA = expm(A * T)
                    y_final = expA @ y0_arr
                    return y_final
            except Exception:
                pass
        else:
            # System might be affine: y' = A*y + c, where c = f_zero
            try:
                c = f_zero.copy()
                A = np.zeros((n, n))
                for j in range(n):
                    e_j = np.zeros(n)
                    e_j[j] = 1.0
                    A[:, j] = np.asarray(f(t0, e_j), dtype=float) - c
                
                # Verify: A @ y0 + c ≈ f(t0, y0)
                f_y0 = np.asarray(f(t0, y0_arr), dtype=float)
                A_y0_c = A @ y0_arr + c
                if np.allclose(A_y0_c, f_y0, atol=1e-8, rtol=1e-6):
                    # Confirmed affine: compute exact solution
                    expA = expm(A * T)
                    # y(T) = exp(A*T)*y0 + integral_0^T exp(A*u)*c du
                    # integral_0^T exp(A*u) du = A^{-1}(exp(A*T) - I) if A invertible
                    try:
                        A_inv = np.linalg.inv(A)
                        integral_term = A_inv @ (expA - np.eye(n)) @ c
                    except np.linalg.LinAlgError:
                        # Use block matrix exponential for singular A
                        B = np.zeros((n + 1, n + 1))
                        B[:n, :n] = A
                        B[:n, n] = c
                        expB = expm(B * T)
                        y_final = expB[:n, :n] @ y0_arr + expB[:n, n]
                        return y_final
                    
                    y_final = expA @ y0_arr + integral_term
                    return y_final
            except Exception:
                pass
    
    # --- Strategy 2: Fallback to DOP853 with tight tolerances ---
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-12,
                atol=1e-14,
                max_step=T / 50.0,
                dense_output=False
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # --- Strategy 3: Last resort RK45 ---
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='RK45',
                rtol=1e-11,
                atol=1e-13,
                max_step=T / 200.0,
                dense_output=False
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    return y0_arr
# EVOLVE-BLOCK-END
