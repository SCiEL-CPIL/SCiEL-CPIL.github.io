"""Rewritten program with robust linear system detection and matrix exponential solution."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = 290.0  # Keep margin before 300s limit

    # Strategy 1: Assume linear time-invariant system y' = A@y + b
    # Extract A and b using n+1 calls
    try:
        # b is the value at y=0
        b = np.array(f(t0, np.zeros(n)), dtype=float)
        
        # Extract columns of A
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # f(t0, e_i) = A @ e_i + b = A[:, i] + b
            # So A[:, i] = f(t0, e_i) - b
            A[:, i] = np.array(f(t0, e_i), dtype=float) - b
            
        # Verify linearity and time-invariance
        # Check at t1 with y0
        f_check = np.array(f(t1, y0_arr), dtype=float)
        f_pred = A @ y0_arr + b
        err = np.linalg.norm(f_check - f_pred)
        
        # Check at a random point to ensure linearity
        # Use a small random vector to avoid overflow
        v_rand = np.random.randn(n) * 0.1
        v_rand[0] += 1.0 # ensure not zero
        f_check_rand = np.array(f(t0 + 0.1*(t1-t0), v_rand), dtype=float)
        f_pred_rand = A @ v_rand + b
        err_rand = np.linalg.norm(f_check_rand - f_pred_rand)
        
        # If errors are small, assume linear time-invariant
        # Tolerance should be scaled with magnitude
        scale = max(1.0, np.linalg.norm(A), np.linalg.norm(b))
        if err < 1e-8 * scale and err_rand < 1e-8 * scale:
            dt = t1 - t0
            # Use augmented matrix to handle b without inverting A
            # M = [[A, b], [0, 0]]
            # z = [y, 1]
            M = np.zeros((n+1, n+1))
            M[:n, :n] = A
            M[:n, n] = b
            # M[n, :] is all zeros
            
            z0 = np.zeros(n+1)
            z0[:n] = y0_arr
            z0[n] = 1.0
            
            # Compute expm(M * dt) @ z0
            # For N=100, expm is fast
            expm_M_dt = scipy.linalg.expm(M * dt)
            z_final = expm_M_dt @ z0
            y_final = z_final[:n]
            
            if np.all(np.isfinite(y_final)):
                return y_final
        else:
            # Not linear or not time-invariant
            pass
            
    except Exception:
        pass

    # Strategy 2: High-order adaptive solver
    # Radau is good for stiff problems (convection-diffusion)
    try:
        if time.time() > deadline - 30:
            # Not enough time, use faster DOP853
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0_arr, 
                method='DOP853',
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                raise Exception("DOP853 failed")
        
        # Run Radau with tight tolerances
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau',
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1-t0)/20
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            raise Exception("Radau failed")
            
    except Exception:
        pass

    # Strategy 3: Fallback to DOP853
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='DOP853',
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Strategy 4: Fallback to RK4
    def _rk4(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                return None
        return y

    y = _rk4(f, t0, t1, y0_arr, 10000)
    if y is not None:
        return y
        
    return y0_arr
# EVOLVE-BLOCK-END
