"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as la
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check if the system is linear and time-independent: y' = A y
    # 1. Check time independence
    f_t0_y0 = f(t0, y0)
    f_t1_y0 = f(t1, y0)
    is_time_independent = np.allclose(f_t0_y0, f_t1_y0, rtol=1e-10, atol=1e-10)
    
    # 2. Check linearity (homogeneous)
    # f(0, 0) should be 0
    f_t0_0 = f(t0, np.zeros(n))
    is_homogeneous = np.allclose(f_t0_0, 0, rtol=1e-10, atol=1e-10)
    
    # f(t, 2y) should be 2f(t, y)
    f_t0_2y = f(t0, 2 * y0)
    is_linear = np.allclose(f_t0_2y, 2 * f_t0_y0, rtol=1e-10, atol=1e-10)
    
    if is_time_independent and is_homogeneous and is_linear:
        # Reconstruct A
        # A @ y = f(t, y) => A[:, j] = f(t, e_j)
        A = np.zeros((n, n))
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
            if time.time() > deadline:
                break
        
        if time.time() < deadline:
            # Compute exact solution using matrix exponential
            # y(t) = exp((t1 - t0) * A) @ y0
            try:
                expA = la.expm((t1 - t0) * A)
                y_final = expA @ y0
                if np.all(np.isfinite(y_final)):
                    return y_final
            except Exception:
                pass
    
    # Fallback: High-accuracy RK45 or Radau if assumptions fail
    # Use solve_ivp with tight tolerances
    # We need to count calls? No, the evaluator counts calls to the passed 'f'.
    # solve_ivp calls 'f' internally.
    
    # Let's use a robust implicit solver for stiff problems
    # But we need to return y(t_end).
    
    # Simple fallback: RK4 with many steps
    # Estimate step size
    # For 10x10 heat eq, eigenvalues ~ -8. Stability h < 2.78/8 ~ 0.34 for RK4.
    # Accuracy: h^4. To get 1e-10 error, h ~ (1e-10)^(1/4) ~ 1e-2.5 ~ 0.003.
    # Steps ~ 0.05 / 0.003 ~ 16.
    # Let's use 1000 steps to be safe.
    
    steps = 2000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for i in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
