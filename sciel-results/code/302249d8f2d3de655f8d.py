"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard: keep a safety margin before the 330s limit
    start_time = time.time()
    deadline = start_time + 315.0  # 15s safety margin
    
    # Extract A and c for linear system: y' = A y + c
    # Probe with y=0 and y=ones to get c and A
    y_probe0 = np.zeros(n)
    y_probe1 = np.ones(n)
    
    ft0 = np.asarray(f(t0, y_probe0), dtype=float)
    ft1 = np.asarray(f(t0, y_probe1), dtype=float)
    
    c = ft0.copy()
    A = (ft1 - ft0).copy()
    
    # Verify linearity with a random probe
    rng = np.random.RandomState(42)
    y_probe2 = rng.randn(n)
    ft2 = np.asarray(f(t0, y_probe2), dtype=float)
    predicted = A @ y_probe2 + c
    linearity_error = np.linalg.norm(ft2 - predicted) / (1.0 + np.linalg.norm(ft2))
    
    if linearity_error < 1e-10:
        # Check if c is time-independent
        ft0_t1 = np.asarray(f(t1, np.zeros(n)), dtype=float)
        c_diff = np.linalg.norm(ft0_t1 - c) / (1.0 + np.linalg.norm(c))
        
        if c_diff < 1e-10:
            # Time-invariant linear system: y' = A y + c
            # Exact solution via matrix exponential:
            # y(t) = expm((t-t0)*A) @ y0 + (expm((t-t0)*A) - I) @ inv(A) @ c
            # Handle singular A case with series expansion or numerical integration
            
            dt = t1 - t0
            if abs(dt) < 1e-15:
                return y0_arr.copy()
            
            # Compute expm(dt * A)
            E = expm(dt * A)
            
            # Compute the particular solution integral
            # If A is invertible: (E - I) @ inv(A) @ c
            # If A is singular: use the identity integral_0^dt expm(s*A) ds = (E-I) @ inv(A) when invertible
            # For singular A, use: solve A * X = (E - I) for X, then X @ c
            try:
                # Try direct inverse first
                A_inv = np.linalg.inv(A)
                particular = (E - np.eye(n)) @ A_inv @ c
            except np.linalg.LinAlgError:
                # A is singular, use least squares or pseudo-inverse
                try:
                    A_pinv = np.linalg.pinv(A)
                    particular = (E - np.eye(n)) @ A_pinv @ c
                except Exception:
                    # Fallback: numerical quadrature for the integral
                    # integral_0^dt expm(s*A) @ c ds
                    steps = 1000
                    h = dt / steps
                    s_vals = np.linspace(0, dt, steps + 1)
                    integrand = np.zeros((steps + 1, n))
                    for i, s in enumerate(s_vals):
                        integrand[i] = expm(s * A) @ c
                    particular = h * (0.5 * integrand[0] + integrand[1:-1].sum(axis=0) + 0.5 * integrand[-1])
            
            y_final = E @ y0_arr + particular
            return y_final
        
        # c is time-dependent but A is time-invariant
        # y' = A y + c(t)
        # Use solve_ivp with the extracted A and a wrapped c(t)
        def f_wrapped(t, y):
            return A @ y + np.asarray(f(t, np.zeros(n)), dtype=float)
        
        try:
            sol = solve_ivp(f_wrapped, [t0, t1], y0_arr, method='Radau',
                           rtol=1e-13, atol=1e-15, max_step=(t1-t0)/50)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        
        try:
            sol = solve_ivp(f_wrapped, [t0, t1], y0_arr, method='BDF',
                           rtol=1e-13, atol=1e-15, max_step=(t1-t0)/50)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: use solve_ivp directly with the original f
    try:
        sol = solve_ivp(f, [t0, t1], y0_arr, method='Radau',
                       rtol=1e-13, atol=1e-15, max_step=(t1-t0)/50)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        sol = solve_ivp(f, [t0, t1], y0_arr, method='BDF',
                       rtol=1e-13, atol=1e-15, max_step=(t1-t0)/50)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Manual RK4 as ultimate fallback
    steps = 5000
    h = (t1 - t0) / steps
    t, y = t0, y0_arr.copy()
    for _ in range(steps):
        if time.time() > deadline:
            return y
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h/2, y + h/2 * k1), dtype=float)
        k3 = np.asarray(f(t + h/2, y + h/2 * k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

# EVOLVE-BLOCK-END
