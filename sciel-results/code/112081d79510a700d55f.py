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
    Detects linear time-invariant system y' = A@y + b and uses matrix exponential.
    Falls back to RK4 if detection fails.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)
    
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 330s kill
    
    # Detect linear time-invariant system: y' = A @ y + b
    # Extract b = f(t0, 0)
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    
    # Check time-invariance: f(t1, 0) should equal f(t0, 0)
    f_at_t1 = f(t1, y_zero)
    if not np.allclose(b, f_at_t1, rtol=1e-9, atol=1e-11):
        return _rk4_fallback(f, t_span, y0)
    
    # Extract A columns: A[:, i] = f(t0, e_i) - b
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i) - b
    
    # Verify linearity with a test point
    y_test = np.ones(n) * 0.5
    f_test = f(t0, y_test)
    f_pred = A @ y_test + b
    if not np.allclose(f_test, f_pred, rtol=1e-7, atol=1e-9):
        return _rk4_fallback(f, t_span, y0)
    
    if time.time() > deadline:
        return _rk4_fallback(f, t_span, y0)
    
    T = t1 - t0
    
    # Compute exp(A*T)
    expAT = expm(A * T)
    
    # Compute integral term: ∫_0^T exp(A*s) b ds
    # Use the formula: if A is invertible, integral = (expAT - I) @ A^{-1} @ b
    # For robustness, use the augmented matrix exponential approach:
    # Create block matrix [[A, b], [0, 0]] of size (n+1)x(n+1)
    # exp([[A,b],[0,0]] * T) = [[exp(A*T), integral], [0, 1]]
    # This avoids conditioning issues with A.
    
    # Build augmented matrix
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    
    # Compute augmented exponential
    exp_aug = expm(A_aug * T)
    
    # Extract results
    integral_term = exp_aug[:n, n]
    y_final = expAT @ y + integral_term
    
    return y_final
# EVOLVE-BLOCK-END
