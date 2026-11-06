"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _extract_linear_operator(f, t, y0):
    """
    Extract the constant matrix A and vector b from y' = A*y + b.
    We need n+1 calls: f(t, y0) gives A*y0 + b.
    Then f(t, y0 + e_i * scale) for each i gives (A*(y0 + e_i*scale) + b).
    So (f(t, y0 + e_i*scale) - f(t, y0)) / scale = A[:, i].
    Total calls: 1 + n = 16 for n=15.
    
    Actually, we can be smarter. Let's use:
    f(t, y) = A @ y + b
    
    Call 1: y_ref = f(t, y0)  -> A @ y0 + b
    Call i+1: y_i = f(t, y0 + delta_i) where delta_i has only component i non-zero
              y_i = A @ (y0 + delta_i) + b = A @ y0 + b + A[:, i] * delta_i[i]
              So A[:, i] = (y_i - y_ref) / delta_i[i]
    
    And b = y_ref - A @ y0
    """
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # Choose a small perturbation scale
    # Use a scale relative to the magnitude of y0
    scale = 1e-8 * np.maximum(1.0, np.abs(y0))
    
    y_ref = f(t, y0)
    
    A = np.empty((n, n), dtype=float)
    for i in range(n):
        perturbed = y0.copy()
        perturbed[i] += scale[i]
        y_pert = f(t, perturbed)
        A[:, i] = (y_pert - y_ref) / scale[i]
    
    b = y_ref - A @ y0
    return A, b


def _matrix_exp_taylor(A, h, K=50):
    """
    Compute exp(A*h) using Taylor series.
    For stiff systems, A*h might have large norm, so we need many terms.
    We'll use scaling and squaring if needed, but for a single step h=1,
    we'll just use a high-order Taylor series.
    
    Actually, for stiff systems with large negative eigenvalues, the Taylor series
    might need many terms. Let's use scaling and squaring:
    exp(A*h) = (exp(A*h/2^s))^(2^s)
    
    But matrix multiplication is expensive. For n=15, it's manageable.
    """
    # Compute norm to determine scaling
    norm_Ah = np.linalg.norm(A * h, ord=np.inf)
    
    # Determine scaling factor s such that norm(A*h/2^s) < 0.25
    s = 0
    if norm_Ah > 0.25:
        s = int(np.ceil(np.log2(norm_Ah / 0.25)))
    
    # Compute exp of the scaled matrix using Taylor series
    B = A * h / (2 ** s)
    
    # Taylor series: sum_{k=0}^{K} B^k / k!
    # Start with identity
    result = np.eye(len(B), dtype=float)
    term = np.eye(len(B), dtype=float)
    
    for k in range(1, K + 1):
        term = term @ B / k
        result = result + term
        # Check convergence
        if np.linalg.norm(term, ord=np.inf) < 1e-16 * np.linalg.norm(result, ord=np.inf):
            break
    
    # Square s times
    for _ in range(s):
        result = result @ result
    
    return result


def integrate(f, t_span, y0):
    """
    Integrate y' = A*y + b from t0 to t1.
    The exact solution is:
    y(t) = exp(A*(t-t0)) * y0 + integral_{t0}^{t} exp(A*(t-s)) * b ds
    
    For constant A and b:
    y(t1) = exp(A*h) * y0 + (exp(A*h) - I) @ A^{-1} * b  (if A is invertible)
    
    More generally, we can compute the integral using:
    integral = exp(A*h) * A^{-1} * (I - exp(-A*h)) * b
    
    But A might be singular. A more robust approach:
    Use the fact that for the ODE y' = A*y + b, the solution is:
    y(t) = exp(A*(t-t0)) * y0 + sum_{k=0}^{inf} A^k * b * (t-t0)^{k+1} / (k+1)!
    
    Actually, let's use:
    y(t1) = exp(A*h) * y0 + (1/h) * integral_0^h exp(A*s) * b ds
    
    We can compute this using the Taylor series approach or by augmenting the system.
    
    A cleaner approach: define the augmented system
    z = [y; 1]
    z' = [A*y + b; 0] = [[A, b], [0, 0]] * z
    
    Then z(t1) = exp([[A, b], [0, 0]] * h) * z0
    
    But this requires computing exp of a (n+1)x(n+1) matrix.
    
    Alternatively, we can use:
    y(t1) = exp(A*h) * y0 + (exp(A*h) - I) @ np.linalg.solve(A, b)  if A invertible
    
    Let's use the augmented matrix approach for robustness.
    """
    t0, t1 = t_span
    h = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract A and b
    A, b = _extract_linear_operator(f, t0, y0)
    
    # Compute exp(A*h)
    exp_Ah = _matrix_exp_taylor(A, h, K=60)
    
    # Compute the integral term: integral_0^h exp(A*s) * b ds
    # This equals (exp(A*h) - I) @ A^{-1} * b if A is invertible
    # For robustness, use the augmented matrix approach
    
    # Augmented matrix M = [[A, b], [0, 0]]
    # But we only need the first n components of exp(M*h) * [y0; 1]
    
    # Let's compute it directly using Taylor series for the integral
    # integral = sum_{k=0}^{K} A^k * b * h^{k+1} / (k+1)!
    
    # This is equivalent to:
    # integral = (exp(A*h) - I) @ A^{-1} @ b  (if A invertible)
    
    # Let's try the direct series approach for the integral
    integral_term = np.zeros(n, dtype=float)
    term = b.copy()  # A^0 * b
    for k in range(1, 61):
        term = term * (h / k)
        integral_term = integral_term + term
        if np.linalg.norm(term) < 1e-16 * np.linalg.norm(integral_term):
            break
    
    # Actually, the series is:
    # integral_0^h exp(A*s) b ds = sum_{k=0}^inf A^k b h^{k+1} / (k+1)!
    # Let's redo this properly
    integral_term = np.zeros(n, dtype=float)
    power = b.copy()  # A^k * b
    for k in range(0, 60):
        # Term: A^k * b * h^{k+1} / (k+1)!
        if k == 0:
            integral_term = integral_term + power * h
        else:
            power = A @ power
            integral_term = integral_term + power * (h ** (k + 1)) / np.math.factorial(k + 1)
        if np.linalg.norm(power * (h ** (k + 1)) / np.math.factorial(k + 1)) < 1e-16:
            break
    
    # This is getting complicated. Let's use the augmented matrix approach.
    # M = [[A, b], [0, 0]] is (n+1) x (n+1)
    # z0 = [y0; 1]
    # z(t1) = exp(M*h) * z0
    # y(t1) = z(t1)[:n]
    
    # Build augmented matrix
    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A
    M[:n, n] = b
    
    # Compute exp(M*h)
    exp_Mh = _matrix_exp_taylor(M, h, K=60)
    
    # z0 = [y0; 1]
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    z1 = exp_Mh @ z0
    y1 = z1[:n]
    
    return y1
# EVOLVE-BLOCK-END
