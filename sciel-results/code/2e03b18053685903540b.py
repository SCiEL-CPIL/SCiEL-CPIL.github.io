import numpy as np
import scipy.linalg as la
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Extract Linear Operator A and Bias b
    # f(t, y) = A y + b
    # b = f(t, 0)
    # A e_i = f(t, e_i) - b
    
    try:
        # Check if t matters. Heat equation is time-invariant, but let's be safe.
        # Usually f(t, y) depends on t only through y for autonomous systems.
        # We assume f(t, y) is linear in y and time-invariant or we just sample at t0.
        # The problem says "fixed ... heat equation", implying autonomous linear system.
        
        zero_vec = np.zeros(n)
        b = np.array(f(t0, zero_vec), dtype=float)
        
        A = np.empty((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # Use t0 for consistency
            A[:, i] = np.array(f(t0, e_i), dtype=float) - b
            
        # Verify linearity with a random vector
        # If f is linear, f(t, y) should equal A @ y + b
        # We use t1 to check if it's time-dependent in a non-linear way or if A changes
        # But for linear time-invariant, A is constant.
        # Let's check with y0 itself at t1? Or a random vector.
        # Let's check f(t1, y0) vs A @ y0 + b.
        f_check = np.array(f(t1, y0), dtype=float)
        pred_check = A @ y0 + b
        error_check = np.linalg.norm(f_check - pred_check)
        scale = max(1.0, np.linalg.norm(f_check))
        
        if error_check > 1e-10 * scale:
            # Not linear or time-varying linear?
            # Fallback to robust ODE solver
            # Use solve_ivp with tight tolerances
            # But solve_ivp calls f. We have already used n+2 calls.
            # We can just use RK4 with many steps or solve_ivp.
            # Given the stiffness of heat equation, Radau or BDF is better.
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: RK4 with very small steps
                # Estimate steps based on error
                return _rk4_fallback(f, t0, t1, y0, 10000)
                
        # Linear Time-Invariant confirmed
        # Compute expm(A * T)
        # Check deadline
        if time.time() > deadline:
             # Should not happen, but just in case
             return y0 # or best guess

        M = la.expm(A * T)
        
        if np.linalg.norm(b) < 1e-15:
            y_final = M @ y0
        else:
            # y(T) = M y0 + v, where v = int_0^T exp(A s) b ds
            # v = A^-1 (M - I) b
            # Solve A v = (M - I) b
            # If A is singular, use pseudo-inverse or handle carefully.
            # Heat equation Laplacian is singular (zero eigenvalue for constant mode).
            # If A is singular, A^-1 doesn't exist.
            # However, if A is singular, (M-I) is also singular in the same null space.
            # We can use least squares or pinv.
            # Or better: solve the augmented system.
            # Actually, for heat equation, the constant mode is preserved.
            # Let's use la.solve with fallback to pinv.
            rhs = (M - np.eye(n)) @ b
            try:
                v = la.solve(A, rhs, assume_a='gen')
            except la.LinAlgError:
                v = la.lstsq(A, rhs)[0]
            y_final = M @ y0 + v
            
        return y_final

    except Exception:
        # Fallback if anything goes wrong
        return _rk4_fallback(f, t0, t1, y0, 10000)

def _rk4_fallback(f, t0, t1, y0, steps):
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
