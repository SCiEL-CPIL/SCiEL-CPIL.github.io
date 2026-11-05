"""ODE solver: convection-diffusion 100-point, using matrix exponential for linear systems."""
import numpy as np
import time
from scipy.linalg import expm


def _detect_linear_tiv(f, t0, t1, y0, n):
    """Detect if f is affine linear and time-invariant. Returns (A, b) or None."""
    # Check time-invariance: evaluate at two different times with same y
    y_test = y0.copy()
    f_at_t0 = f(t0, y_test)
    f_at_t1 = f(t1, y_test)
    if not np.allclose(f_at_t0, f_at_t1, rtol=1e-10, atol=1e-12):
        # Also try midpoint
        f_at_mid = f((t0 + t1) / 2, y_test)
        if not np.allclose(f_at_t0, f_at_mid, rtol=1e-10, atol=1e-12):
            return None
    
    # Determine b: evaluate at y=0
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    
    # Determine A: evaluate at standard basis vectors
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
    
    # Verify linearity: check f(t, y0) == A @ y0 + b
    predicted = A @ y0 + b
    actual = f(t0, y0)
    if not np.allclose(predicted, actual, rtol=1e-8, atol=1e-10):
        return None
    
    return A, b


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=100000):
    """Adaptive RK45 (Dormand-Prince) method."""
    t = t0
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Dormand-Prince coefficients
    a2, a3, a4, a5, a6, a7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    b2_1 = 1/5
    b3_1, b3_2 = 3/40, 9/40
    b4_1, b4_2, b4_3 = 44/45, -56/15, 32/9
    b5_1, b5_2, b5_3, b5_4 = -11/54, 5/2, 70/27, 35/27
    b6_1, b6_2, b6_3, b6_4, b6_5 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096
    b7_1, b7_2, b7_3, b7_4, b7_5, b7_6 = -17253/339200, 2245/19200, 6839/110592, 281/560, -1105/17280, 41/840
    
    # Error coefficients (5th order)
    e2 = 1/40
    e3 = 3/40 - 9/40
    e4 = 44/45 - 56/15 + 32/9 - 1901/79200
    e5 = -11/54 + 5/2 - 70/27 + 35/27 - 241/720
    e6 = 1631/55296 - 175/512 + 575/13824 - 44275/110592 + 253/4096 - 109/1440
    e7 = -17253/339200 + 2245/19200 + 6839/110592 + 281/560 - 1105/17280 + 41/840
    
    h = (t1 - t0) / 100  # initial step
    h = max(h, 1e-15)
    
    for _ in range(max_steps):
        if t >= t1 - 1e-15:
            break
        
        # Adjust step to hit t1 exactly
        if t + h > t1:
            h = t1 - t
        
        k1 = f(t, y)
        k2 = f(t + a2 * h, y + h * b2_1 * k1)
        k3 = f(t + a3 * h, y + h * (b3_1 * k1 + b3_2 * k2))
        k4 = f(t + a4 * h, y + h * (b4_1 * k1 + b4_2 * k2 + b4_3 * k3))
        k5 = f(t + a5 * h, y + h * (b5_1 * k1 + b5_2 * k2 + b5_3 * k3 + b5_4 * k4))
        k6 = f(t + a6 * h, y + h * (b6_1 * k1 + b6_2 * k2 + b6_3 * k3 + b6_4 * k4 + b6_5 * k5))
        
        # 4th order solution
        y_new = y + h * (35/384 * k1 + 0 * k2 + 500/1113 * k3 + 125/192 * k4 + 2187/6784 * k5 + 11/84 * k6)
        
        # Error estimate (5th order - 4th order)
        err = h * (e2 * k1 + e3 * k2 + e4 * k3 + e5 * k4 + e6 * k5 + e7 * k6)
        err_norm = np.linalg.norm(err) / (atol + rtol * np.linalg.norm(y_new))
        
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        if err_norm <= 1.0:
            t += h
            y = y_new
            # Adjust step for next
            if err_norm > 1e-8:
                h *= min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                h *= 2.0
            h = min(h, t1 - t)
        else:
            h *= max(0.1, 0.9 * err_norm ** (-0.25))
    
    return y


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Try to detect linear time-invariant system and use exact solution
    try:
        Ab = _detect_linear_tiv(f, t0, t1, y0, n)
        if Ab is not None:
            A, b = Ab
            if time.time() > deadline:
                return y0
            
            # Compute exact solution: y(t) = e^{At} y0 + integral_0^t e^{A(t-s)} b ds
            # For constant b: y(t) = e^{At} y0 + (e^{At} - I) A^{-1} b (if A invertible)
            # More robust: augment system
            # Use the formula: y(t) = e^{At} y0 + int_0^t e^{A(t-s)} b ds
            # = e^{At} y0 + e^{At} int_0^t e^{-As} b ds
            # Let u(t) = int_0^t e^{-As} b ds, then u' = -A u + b, u(0) = 0
            # So [y; u]' = [[A, 0], [-A, I]] [y; u] + [0; b] ... this gets complicated
            # Simpler: just use expm directly
            t_end = t1 - t0
            expA = expm(A * t_end)
            
            # For the integral term, use: int_0^t e^{A(t-s)} b ds
            # This equals (e^{At} - I) A^{-1} b if A is invertible
            # Or use the augmented matrix approach
            # Most robust: use scipy's odeint or just numerical quadrature with expm
            # Actually, let's use: y(t) = e^{At} y0 + sum using the fact that
            # int_0^t e^{A(t-s)} b ds can be computed as:
            # Create augmented system or use: (e^{At} - I) A^{-1} b with care
            
            try:
                # Try direct formula
                A_inv_b = np.linalg.solve(A, b)
                y_final = expA @ y0 + (expA - np.eye(n)) @ A_inv_b
            except np.linalg.LinAlgError:
                # A is singular, use augmented system
                # y(t) = e^{At} y0 + int_0^t e^{A(t-s)} b ds
                # Use numerical integration with expm at a few points (Simpson's rule)
                n_quad = 100
                ts = np.linspace(0, t_end, n_quad + 1)
                hs = ts[1] - ts[0]
                integrand = np.zeros((n_quad + 1, n))
                for i, s in enumerate(ts):
                    expA_s = expm(A * (t_end - s))
                    integrand[i] = expA_s @ b
                # Simpson's rule
                integral = np.zeros(n)
                for i in range(n):
                    integral[i] = (hs / 3) * (integrand[0, i] + integrand[-1, i] +
                        4 * np.sum(integrand[1:-1:2, i]) + 2 * np.sum(integrand[2:-1:2, i]))
                y_final = expA @ y0 + integral
            
            if not np.all(np.isfinite(y_final)):
                raise ValueError("Non-finite result from matrix exponential")
            
            return y_final
    except Exception:
        pass
    
    # Fallback: adaptive RK45
    if time.time() > deadline:
        return y0
    
    y = _rk45_adaptive(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=50000)
    
    if not np.all(np.isfinite(y)):
        # Last resort: fixed-step RK4
        steps = 5000
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
    
    return y
