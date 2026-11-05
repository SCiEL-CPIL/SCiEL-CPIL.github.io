import numpy as np
import time
import scipy.linalg as la
import scipy.integrate as spi

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Detect if the system is linear (y' = A(t)y) and time-invariant.
    2. If so, extract the matrix A and use the matrix exponential for an exact solution.
    3. If not, fallback to a high-accuracy adaptive Runge-Kutta solver.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Attempt Linear Operator Extraction
    # Hypothesis: y' = A y
    # We need to find A.
    # Probe f at t0 using basis vectors.
    
    A = np.zeros((n, n))
    
    # To save time and calls, we can try to infer A from a few random probes or basis vectors.
    # Basis vectors require n calls. n=100 is very cheap.
    
    try:
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # Evaluate at t0
            val = f(t0, e_i)
            A[:, i] = val
            
        # Check linearity: f(t, c1*y1 + c2*y2) == c1*f(t,y1) + c2*f(t,y2)
        # And time-invariance: A(t0) == A(t1)
        
        # Check time invariance with one probe
        t_mid = 0.5 * (t0 + t1)
        e_test = np.ones(n) / np.sqrt(n)  # Random-ish vector
        val_mid = f(t_mid, e_test)
        # If A is constant, f(t_mid, e_test) should be A @ e_test
        predicted = A @ e_test
        diff = np.linalg.norm(val_mid - predicted)
        scale = np.linalg.norm(predicted) + 1e-12
        rel_diff = diff / scale
        
        # Also check linearity with a simple 2-vector test
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)
        c1, c2 = 0.3, 0.7
        v_test = c1 * v1 + c2 * v2
        f_test = f(t0, v_test)
        f_linear = c1 * f(t0, v1) + c2 * f(t0, v2)
        lin_diff = np.linalg.norm(f_test - f_linear)
        lin_scale = np.linalg.norm(f_linear) + 1e-12
        lin_rel = lin_diff / lin_scale
        
        # If it's linear and time-invariant
        if rel_diff < 1e-10 and lin_rel < 1e-10:
            # Use matrix exponential
            # y(t1) = exp((t1-t0) * A) @ y0
            dt = t1 - t0
            M = la.expm(dt * A)
            y_final = M @ y0
            
            if np.all(np.isfinite(y_final)):
                return y_final
        else:
            # System is not simply linear time-invariant y'=Ay
            # It might have a source term y' = Ay + b(t)
            # Or be non-linear.
            # For convection-diffusion, it's usually linear. 
            # If it failed, maybe it's time-dependent A(t).
            # We can try to integrate using the extracted A if it varies slowly? 
            # Or just fallback to robust solver.
            pass
            
    except Exception as e:
        # If extraction fails, fallback
        pass
    
    # Fallback: High-accuracy adaptive solver
    # Use Radau or BDF if stiff, RK45 if not.
    # Given the previous attempts, Radau with tight tol is good.
    
    # Try a few methods to ensure success
    methods = [
        ('Radau', {'rtol': 1e-13, 'atol': 1e-15}),
        ('LSODA', {'rtol': 1e-13, 'atol': 1e-15}),
        ('RK45', {'rtol': 1e-13, 'atol': 1e-15})
    ]
    
    for method_name, opts in methods:
        if time.time() > deadline:
            break
        try:
            if method_name not in spi.solve_ivp.methods:
                continue
            sol = spi.solve_ivp(
                f, t_span, y0,
                method=method_name,
                **opts
            )
            if sol.success:
                y_final = sol.y[:, -1]
                if np.all(np.isfinite(y_final)):
                    return y_final
        except Exception:
            pass
            
    # Last resort: Fixed step RK4
    steps = 10000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or time.time() > deadline:
            break
            
    return y
