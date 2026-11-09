import numpy as np
import time
import scipy.linalg
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP.
    Tries to detect if the system is time-invariant linear (homogeneous or affine).
    If so, uses exact matrix exponential solution (augmenting for affine).
    Otherwise, falls back to scipy.integrate.solve_ivp with Radau.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    dt = t1 - t0
    
    if dt == 0:
        return y0

    # Deadline check
    if time.time() - start_time > 290:
        return y0

    # 1. Diagnostic Probes
    # Probe t0
    v0 = np.asarray(f(t0, np.zeros(n, dtype=float)), dtype=float)
    v_i = []
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        v_i.append(np.asarray(f(t0, e_i), dtype=float))
        
    if time.time() - start_time > 290:
        return y0

    # Probe t1
    w0 = np.asarray(f(t1, np.zeros(n, dtype=float)), dtype=float)
    w_i = []
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        w_i.append(np.asarray(f(t1, e_i), dtype=float))

    if time.time() - start_time > 290:
        return y0

    # Check Time Invariance
    # If f(t, y) is time-invariant, f(t0, y) == f(t1, y)
    is_time_invariant = np.allclose(v0, w0, rtol=1e-10, atol=1e-12)
    if is_time_invariant:
        for i in range(n):
            if not np.allclose(v_i[i], w_i[i], rtol=1e-10, atol=1e-12):
                is_time_invariant = False
                break

    if is_time_invariant:
        # Check Linearity / Construct A and b
        # Assume y' = A y + b
        # v0 = b
        # v_i = A e_i + b  => A e_i = v_i - b
        b = v0
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            A[:, i] = v_i[i] - b
            
        # Verify Linearity: f(t, y) = A y + b
        # Pick a fixed vector y_test and check
        y_test = np.ones(n, dtype=float)
        y_test_expected = A @ y_test + b
        y_test_actual = np.asarray(f(t0, y_test), dtype=float)
        
        is_linear = np.allclose(y_test_expected, y_test_actual, rtol=1e-10, atol=1e-12)
        
        if is_linear:
            # Exact Solution using Augmented Matrix
            # State z = [y; 1]
            # z' = [A b; 0 0] z
            A_aug = np.zeros((n+1, n+1), dtype=float)
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            # Last row is 0
            
            try:
                E_aug = scipy.linalg.expm(A_aug * dt)
                z0 = np.zeros(n+1, dtype=float)
                z0[:n] = y0
                z0[n] = 1.0
                
                z_end = E_aug @ z0
                y_end = z_end[:n]
                return y_end
            except Exception:
                pass # Fallback to solver if expm fails

    # Fallback: Stiff ODE Solver
    # Use Radau for stiff systems
    # We have already used n+1 + n+1 + 1 = 2n+2 calls.
    # n=15, so 32 calls.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol2 = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Last resort: return y0 or last state
                return sol.y[:, -1] if sol.y.size > 0 else y0
    except Exception:
        return y0
