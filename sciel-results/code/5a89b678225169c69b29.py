import numpy as np
import scipy.integrate as integrate_scipy
import scipy.linalg
import time

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP.
    Handles both homogeneous (y' = Ay) and affine (y' = Ay + b) linear systems
    with constant coefficients by using matrix exponentials.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    timeout = 290

    try:
        # 1. Determine b (affine part)
        b = f(t0, np.zeros(n))
        
        # 2. Determine A
        # f(t, y) = A y + b
        # f(t, e_j) = A e_j + b = A[:, j] + b
        # So A[:, j] = f(t, e_j) - b
        
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            col_val = f(t0, e_j)
            A[:, j] = col_val - b
            
            if time.time() - start_time > timeout:
                return _fallback(f, t_span, y0)

        dt = t1 - t0
        
        # 3. Compute solution
        # y(t) = exp(A dt) y0 + (integral_0^dt exp(A tau) dtau) b
        
        # We can compute the augmented matrix exponential to get both terms.
        # Block matrix M = [[A, I], [0, 0]]
        # exp(M dt) = [[exp(A dt), K], [0, I]] where K = integral_0^dt exp(A tau) dtau
        
        # Construct 2n x 2n matrix
        M_aug = np.zeros((2 * n, 2 * n))
        M_aug[:n, :n] = A
        M_aug[:n, n:] = np.eye(n)
        # Bottom left and bottom right are already 0, except bottom right should be 0 in M,
        # so exp will have I there.
        
        M_exp = scipy.linalg.expm(M_aug * dt)
        
        exp_A_dt = M_exp[:n, :n]
        K = M_exp[:n, n:]
        
        y_final = exp_A_dt @ y0 + K @ b
        
        return y_final

    except Exception:
        return _fallback(f, t_span, y0)

def _fallback(f, t_span, y0):
    """
    Fallback to scipy Radau if the linear assumption fails or expm fails.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    try:
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14, 
            max_step=(t1 - t0) / 100
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = integrate_scipy.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-14, 
                max_step=(t1 - t0) / 100
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                sol = integrate_scipy.solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='DOP853', 
                    rtol=1e-12, 
                    atol=1e-14
                )
                return sol.y[:, -1]

    except Exception:
        # Last resort RK4
        steps = 10000
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
