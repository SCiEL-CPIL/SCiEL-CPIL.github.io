import numpy as np
import time
import scipy.linalg as la
import scipy.integrate as spi

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Detect if the system is linear/affine and time-invariant (y' = A y + b).
    2. If so, extract A and b and use the matrix exponential of the augmented system for an exact solution.
    3. If not, fallback to a high-accuracy adaptive Runge-Kutta solver.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # --- Attempt Linear/Affine Time-Invariant Extraction ---
    
    # Hypothesis: y' = A y + b
    # 1. Probe b = f(t0, 0)
    try:
        # Probe at t0
        b = np.asarray(f(t0, np.zeros(n)), dtype=float)
        
        # 2. Probe A columns: A[:, i] = f(t0, e_i) - b
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            val = np.asarray(f(t0, e_i), dtype=float)
            A[:, i] = val - b
            
        # 3. Verify time-invariance and affine structure
        # Use a couple of random test points
        np.random.seed(42) # Deterministic for verification stability? No, random is fine for check.
        y_test1 = np.random.rand(n)
        y_test2 = np.random.rand(n)
        
        t_test1 = 0.25 * (t1 - t0) + t0
        t_test2 = 0.75 * (t1 - t0) + t0
        
        # Check 1
        pred1 = A @ y_test1 + b
        act1 = np.asarray(f(t_test1, y_test1), dtype=float)
        
        # Check 2
        pred2 = A @ y_test2 + b
        act2 = np.asarray(f(t_test2, y_test2), dtype=float)
        
        err1 = np.linalg.norm(act1 - pred1)
        err2 = np.linalg.norm(act2 - pred2)
        
        scale1 = np.linalg.norm(pred1) + 1e-12
        scale2 = np.linalg.norm(pred2) + 1e-12
        
        rel_err1 = err1 / scale1
        rel_err2 = err2 / scale2
        
        # If errors are very small, we assume it's affine time-invariant
        if rel_err1 < 1e-10 and rel_err2 < 1e-10:
            dt = t1 - t0
            
            # Construct augmented matrix M for the system:
            # z' = [ A  b ] [ z ]
            #     [ 0  1 ] [ 1 ]
            # where z = [y; 1]
            
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A
            M[:n, n] = b
            M[n, n] = 1.0
            
            # Initial state for augmented system
            z0 = np.zeros(n + 1)
            z0[:n] = y0
            z0[n] = 1.0
            
            # Compute matrix exponential
            # expm is robust for these sizes
            ExpM = la.expm(dt * M)
            
            z_final = ExpM @ z0
            y_final = z_final[:n]
            
            if np.all(np.isfinite(y_final)):
                return y_final
        else:
            # System is not affine time-invariant
            # Fallback to numerical integration
            pass
            
    except Exception:
        # If extraction fails, fallback
        pass

    # --- Fallback: High-accuracy adaptive solver ---
    
    # Diffusion problems are typically stiff. Explicit methods like DOP853 may require
    # very small steps, leading to high function call counts. Implicit methods like
    # Radau or BDF are better suited for stiff systems and generally have lower call counts
    # for the same accuracy in stiff regimes.
    
    methods = [
        ('Radau', {'rtol': 1e-10, 'atol': 1e-12}),
        ('BDF', {'rtol': 1e-10, 'atol': 1e-12}),
        ('DOP853', {'rtol': 1e-12, 'atol': 1e-14}),
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
    # This is a safety net, likely to be inaccurate or slow, but prevents crash.
    # Use 5000 steps as a balance between accuracy and speed.
    steps = 5000
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
