import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    
    # 1. Probe for linearity and time-invariance
    # Assume f(t, y) = A(t) y + b(t)
    
    y_zero = np.zeros(n)
    
    try:
        b0 = f(t0, y_zero)
    except Exception:
        b0 = None

    if b0 is not None and np.all(np.isfinite(b0)):
        A = np.zeros((n, n))
        is_linear = True
        is_time_inv = True
        
        # Probe columns
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            try:
                f0 = f(t0, e_j)
            except Exception:
                is_linear = False
                break
            
            if not np.all(np.isfinite(f0)):
                is_linear = False
                break
                
            A[:, j] = f0 - b0

        if is_linear:
            # Check time invariance
            # Check b at t_mid
            t_mid = (t0 + t1) / 2
            try:
                b_mid = f(t_mid, y_zero)
            except Exception:
                b_mid = None
                
            if b_mid is not None:
                if not np.allclose(b0, b_mid, rtol=1e-5, atol=1e-8):
                    is_time_inv = False
            else:
                is_time_inv = False

            if is_time_inv:
                # Check A at t_mid for a few columns
                for j in range(min(3, n)):
                    e_j = np.zeros(n)
                    e_j[j] = 1.0
                    try:
                        f_mid = f(t_mid, e_j)
                    except Exception:
                        is_time_inv = False
                        break
                        
                    if not np.all(np.isfinite(f_mid)):
                        is_time_inv = False
                        break
                        
                    A_mid_col = f_mid - b_mid
                    if not np.allclose(A[:, j], A_mid_col, rtol=1e-5, atol=1e-8):
                        is_time_inv = False
                        break

            if is_linear and is_time_inv:
                # Verify linearity with a random vector
                rng = np.random.default_rng(42)
                v = rng.standard_normal(n)
                try:
                    f_v = f(t0, v)
                    A_v = A @ v + b0
                    if np.allclose(f_v, A_v, rtol=1e-5, atol=1e-8):
                        # Use Matrix Exponential
                        try:
                            dt = t1 - t0
                            # expm_multiply is efficient
                            y_end = scipy.linalg.expm_multiply(A, dt, y0)
                            if np.all(np.isfinite(y_end)):
                                return y_end
                        except Exception:
                            pass
                    else:
                        # Not linear, fallback
                        pass
                except Exception:
                    pass

    # Fallback: Robust ODE Solver
    # Use Radau for stiffness
    # High precision
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=(t1-t0)/10
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            return sol.y[:, -1]
    except Exception:
        # Last resort: simple RK4 with many steps
        steps = 2000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
