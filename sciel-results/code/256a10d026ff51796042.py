"""ODE solver for convection-diffusion 100-point IVP using matrix exponential."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm, solve


def _rk4_fallback(f, t_span, y0, max_calls=2000000):
    """Fixed-step RK4 fallback if matrix approach fails."""
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)
    
    # Estimate step size to stay within call budget
    # RK4 uses 4 calls per step
    max_steps = max_calls // 4
    h = (t1 - t0) / max_steps
    
    t = t0
    calls = 0
    while t < t1 and calls < max_calls:
        if t + h > t1:
            h = t1 - t
        k1 = f(t, y)
        calls += 1
        k2 = f(t + h/2, y + h/2 * k1)
        calls += 1
        k3 = f(t + h/2, y + h/2 * k2)
        calls += 1
        k4 = f(t + h, y + h * k3)
        calls += 1
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    Attempts to detect linear time-invariant system y' = A@y + b and use matrix exponential.
    Falls back to RK4 if detection fails.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Try to detect linear time-invariant system: y' = A @ y + b
    # Strategy: call f at two points to extract A and b
    # f(t, y) = A @ y + b (if linear and time-invariant)
    
    # First, check if f is time-invariant by calling at same y, different t
    y_test = np.zeros(n)
    f_at_t0 = f(t0, y_test)
    f_at_tmid = f((t0 + t1) / 2, y_test)
    
    # Check time invariance
    if not np.allclose(f_at_t0, f_at_tmid, rtol=1e-10, atol=1e-12):
        # Not time-invariant, use fallback
        return _rk4_fallback(f, t_span, y0)
    
    b = f_at_t0  # f(t, 0) = b
    
    # Extract A by calling f at unit vectors
    # f(t, e_i) = A @ e_i + b = column i of A + b
    # So column i of A = f(t, e_i) - b
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_ei = f(t0, e_i)
        A[:, i] = f_ei - b
    
    # Verify linearity: check that f(t, y) = A @ y + b for a random test point
    y_random = np.random.randn(n) * 0.1
    f_random = f(t0, y_random)
    f_predicted = A @ y_random + b
    if not np.allclose(f_random, f_predicted, rtol=1e-8, atol=1e-10):
        # Not linear, use fallback
        return _rk4_fallback(f, t_span, y0)
    
    # Check deadline
    if time.time() > deadline:
        return _rk4_fallback(f, t_span, y0)
    
    # Now solve y' = A @ y + b using matrix exponential
    # Solution: y(t) = e^{A(t-t0)} @ y0 + integral_0^{t-t0} e^{A s} b ds
    # For constant b: y(t) = e^{A T} @ y0 + (e^{A T} - I) @ A^{-1} @ b if A is invertible
    # More robust: use the formula y(T) = e^{AT} y0 + ∫_0^T e^{As} b ds
    # The integral can be computed as: (e^{AT} - I) @ A^{-1} @ b if A is invertible
    # Or more generally using the augmented system or numerical quadrature of the matrix exponential
    
    T = t1 - t0
    
    # Compute e^{A T}
    expAT = expm(A * T)
    
    # Compute the integral term: ∫_0^T e^{A s} b ds
    # If A is invertible: (e^{AT} - I) @ inv(A) @ b
    # Use solve instead of inv for numerical stability
    try:
        # Check if A is close to singular
        cond_A = np.linalg.cond(A)
        if cond_A < 1e12:
            # A is well-conditioned, use the formula
            integral_term = solve(A, (expAT - np.eye(n)) @ b)
        else:
            # Use augmented system approach
            # Create augmented matrix [A, b; 0, 0] and compute its exponential
            # Or use numerical integration
            # Use the identity: ∫_0^T e^{As} b ds = [e^{AT} - I] A^{-1} b
            # For ill-conditioned A, use SVD-based pseudo-inverse or direct quadrature
            # Use Simpson's rule with matrix exponentials
            n_quad = 100
            s_vals = np.linspace(0, T, n_quad + 1)
            ds = T / n_quad
            integral = np.zeros(n)
            for j, s in enumerate(s_vals):
                weight = 1 if j == 0 or j == n_quad else (4 if j % 2 == 1 else 2)
                integral += weight * expm(A * s) @ b
            integral_term = integral * ds / 3
    except np.linalg.LinAlgError:
        # Use numerical quadrature
        n_quad = 100
        s_vals = np.linspace(0, T, n_quad + 1)
        ds = T / n_quad
        integral = np.zeros(n)
        for j, s in enumerate(s_vals):
            weight = 1 if j == 0 or j == n_quad else (4 if j % 2 == 1 else 2)
            integral += weight * expm(A * s) @ b
        integral_term = integral * ds / 3
    
    y_final = expAT @ y + integral_term
    
    # Check deadline
    if time.time() > deadline:
        return y_final
    
    return y_final
# EVOLVE-BLOCK-END
