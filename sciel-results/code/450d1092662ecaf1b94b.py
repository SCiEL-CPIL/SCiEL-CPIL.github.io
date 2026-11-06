"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Detect if the system is linear and time-invariant: f(t, y) = A y + b.
    2. Reconstruct A and b from evaluations.
    3. Compute the exact solution using Padé approximation of the matrix exponential
       applied to the augmented system, or directly to A with the affine correction.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    dt_total = t1 - t0
    
    if dt_total == 0:
        return y0_arr

    # Step 1: Check for affine linear time-invariant system: f(t, y) = A y + b
    
    # Estimate b = f(t0, 0)
    y_zero = np.zeros_like(y0_arr)
    b = f(t0, y_zero)
    
    # Reconstruct A from basis vectors at t0
    # A @ e_j = f(t0, e_j) - b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros_like(y0_arr)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Verify linearity and time-invariance
    # Check 1: Does A y0 + b match f(t0, y0)?
    pred_t0 = A @ y0_arr + b
    actual_t0 = f(t0, y0_arr)
    err_t0 = np.linalg.norm(pred_t0 - actual_t0)
    
    # Check 2: Does A y0 + b match f(t1, y0)?
    actual_t1 = f(t1, y0_arr)
    err_t1 = np.linalg.norm(pred_t0 - actual_t1)
    
    scale = max(1.0, np.linalg.norm(actual_t0))
    
    if err_t0 < 1e-8 * scale and err_t1 < 1e-8 * scale:
        # System is affine linear and time-invariant: y' = A y + b
        # Use Padé approximation for the matrix exponential
        
        # For y' = A y + b, the solution is:
        # y(t) = exp(A(t-t0)) y0 + ∫_0^{t-t0} exp(A τ) dτ b
        # 
        # We can compute this using the augmented system or by:
        # 1. Computing E = exp(A * dt)
        # 2. Computing the integral term: if A is invertible, it's A^{-1}(E - I) b
        #    Otherwise, we can use the augmented matrix approach
        #
        # Let's use the augmented matrix approach for generality:
        # z = [y; 1], z' = [[A, b], [0, 0]] z
        # z(t1) = exp(M * dt) z(t0) where M = [[A, b], [0, 0]]
        
        # Build augmented matrix M (n+1 x n+1)
        M = np.zeros((n + 1, n + 1), dtype=float)
        M[:n, :n] = A
        M[:n, n] = b
        # Last row is already zeros
        
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0_arr
        z0[n] = 1.0
        
        # Compute exp(M * dt) using Padé approximation
        E = _pade_expm(M * dt_total)
        z1 = E @ z0
        
        return z1[:n]
    else:
        # Fallback: Use RK45 or similar
        # For robustness, use a high-order RK method
        steps = 10000
        h = dt_total / steps
        t = t0
        y = y0_arr.copy()
        
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            
        return y


def _pade_expm(B, order=13):
    """
    Compute exp(B) using a Padé approximation of the given order.
    Uses the [p/p] Padé approximant: R_p(B) = N_p(B) / D_p(B)
    where N_p and D_p are polynomials in B.
    
    For order p, the Padé approximant is:
    R_p(B) = (sum_{k=0}^{p} a_k B^k) / (sum_{k=0}^{p} (-1)^k a_k B^k)
    
    where a_k = (2p-k)! * p! / (k! * (p-k)! * (2p)!)
    
    We use scaling and squaring to ensure convergence.
    """
    n = B.shape[0]
    
    # Determine scaling factor: choose s such that ||B / 2^s||_1 <= theta_p
    # For p=13, theta_13 ≈ 3.75
    # For p=7, theta_7 ≈ 1.4
    # We'll use p=13 for high accuracy
    
    theta = 3.75  # threshold for p=13
    norm_B = np.linalg.norm(B, 1)
    
    if norm_B == 0:
        return np.eye(n)
    
    # Choose s such that norm_B / 2^s <= theta
    s = 0
    while norm_B / (2 ** s) > theta:
        s += 1
    
    # Scale B
    B_scaled = B / (2 ** s)
    
    # Compute Padé approximant R_p(B_scaled)
    p = order
    a = _pade_coefficients(p)
    
    # Compute N = sum_{k=0}^{p} a_k B^k and D = sum_{k=0}^{p} (-1)^k a_k B^k
    I = np.eye(n)
    Bk = I.copy()
    N = a[0] * I
    D = a[0] * I
    
    for k in range(1, p + 1):
        Bk = Bk @ B_scaled
        N += a[k] * Bk
        D += ((-1.0) ** k) * a[k] * Bk
    
    # R_p(B) = N @ inv(D)
    # Use solve for better numerical stability
    R = np.linalg.solve(D.T, N.T).T
    
    # Squaring: exp(B) = R^(2^s)
    result = R
    for _ in range(s):
        result = result @ result
    
    return result


def _pade_coefficients(p):
    """
    Compute the coefficients a_k for the [p/p] Padé approximant to the exponential.
    a_k = (2p-k)! * p! / (k! * (p-k)! * (2p)!)
    """
    from math import factorial
    a = np.zeros(p + 1)
    fact_2p = factorial(2 * p)
    for k in range(p + 1):
        a[k] = (factorial(2 * p - k) * factorial(p)) / (factorial(k) * factorial(p - k) * fact_2p)
    return a

# EVOLVE-BLOCK-END
