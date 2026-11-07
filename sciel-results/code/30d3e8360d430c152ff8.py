"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial Hn(x) as a list [c0, c1, ..., cn] for x^0, x^1, ..."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    coeffs = [0.0] * (n + 1)
    # H_{n+1} = 2x*H_n - 2n*H_{n-1}
    prev = [0.0] * n  # H_{n-1}
    curr = [0.0] * (n)  # H_n
    # Base
    prev[0] = 1.0  # H0
    curr[0] = 0.0
    curr[1] = 2.0  # H1
    if n == 0:
        return prev
    if n == 1:
        return curr
    for k in range(2, n + 1):
        # H_k = 2x*H_{k-1} - 2*(k-1)*H_{k-2}
        new = [0.0] * (k + 1)
        # 2x * H_{k-1}: shift curr by 1, multiply by 2
        for i in range(k):
            new[i + 1] += 2.0 * curr[i]
        # -2*(k-1) * H_{k-2}
        for i in range(k - 1):
            new[i] -= 2.0 * (k - 1) * prev[i]
        prev = curr
        curr = new
    return curr

def _compute_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients."""
    # Get coefficients of H0, H4, H8, H12
    H0 = _hermite_coeffs(0)   # [1]
    H4 = _hermite_coeffs(4)   # degree 4
    H8 = _hermite_coeffs(8)   # degree 8
    H12 = _hermite_coeffs(12) # degree 12
    
    # Build g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # g is a polynomial of degree 12 (even powers only)
    # Represent as array of coefficients for x^0, x^2, x^4, ..., x^12 (7 terms)
    
    def poly_add_even(coeffs_list, scalar):
        """Add scalar * coeffs_list to an even-power polynomial represented as list."""
        pass
    
    # Let's represent polynomials in terms of y = x^2, so degree 6 in y
    # H0(y) = 1
    # H4(y) = 16y^2 - 48y + 12
    # H8(y) = 256y^4 - 3072y^3 + 11520y^2 - 13440y + 1680
    # H12(y) = 4096y^6 - 73728y^5 + 524280y^4 - 1468800y^3 + 1728960y^2 - 725760y + 103950
    
    # Convert to y-polynomials (degree 6 max)
    def to_y_poly(h_coeffs):
        """Convert x-polynomial coefficients to y=x^2 polynomial coefficients."""
        # h_coeffs[i] is coefficient of x^i
        # Only even powers should be nonzero
        y_coeffs = [0.0] * 7
        for i in range(0, len(h_coeffs), 2):
            if i // 2 < 7:
                y_coeffs[i // 2] = h_coeffs[i]
        return y_coeffs
    
    H0_y = to_y_poly(H0)
    H4_y = to_y_poly(H4)
    H8_y = to_y_poly(H8)
    H12_y = to_y_poly(H12)
    
    # g_y = c0*H0_y + c1*H4_y + c2*H8_y + c3*H12_y
    # g_y(0) = 0 => c0*H0_y[0] + c1*H4_y[0] + c2*H8_y[0] + c3*H12_y[0] = 0
    g0 = c0 * H0_y[0] + c1 * H4_y[0] + c2 * H8_y[0]
    c3 = -g0 / H12_y[0] if H12_y[0] != 0 else 0.0
    
    g_y = [0.0] * 7
    for i in range(7):
        g_y[i] = c0 * H0_y[i] + c1 * H4_y[i] + c2 * H8_y[i] + c3 * H12_y[i]
    
    # q(y) = g(x)/x^2 = g_y(y) / y, where y = x^2
    # q(y) = g_y[0]/y + g_y[1] + g_y[2]*y + ... + g_y[6]*y^5
    # But g_y[0] should be 0 (since g(0)=0), so q(y) = g_y[1] + g_y[2]*y + ... + g_y[6]*y^5
    # This is a polynomial of degree 5 in y
    
    # Check: g_y[0] should be ~0
    # q(y) = sum_{i=1}^{6} g_y[i] * y^(i-1)
    q_coeffs = [g_y[i] for i in range(1, 7)]  # degree 5 in y
    
    # Find real roots of q(y) = 0 where y >= 0
    # Use numpy roots
    if all(abs(c) < 1e-15 for c in q_coeffs):
        return float('inf')
    
    # Find roots of q(y)
    roots = np.roots(q_coeffs)
    
    # Filter real roots with y >= 0
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            ry = r.real
            if ry >= -1e-10:
                real_roots.append(max(0.0, ry))
    
    if not real_roots:
        return float('inf')
    
    # For each real root y_r, check if q changes sign at y_r
    # q changes sign at y_r if the multiplicity is odd
    # We can check by evaluating q at y_r - eps and y_r + eps
    
    sign_change_roots = []
    eps = 1e-8
    
    def q_eval(y):
        """Evaluate q(y) = sum q_coeffs[i] * y^i"""
        result = 0.0
        for i in range(len(q_coeffs)):
            result += q_coeffs[i] * (y ** i)
        return result
    
    for y_r in real_roots:
        # Check sign change
        y_left = max(0.0, y_r - eps)
        y_right = y_r + eps
        q_left = q_eval(y_left)
        q_right = q_eval(y_right)
        
        # Sign change: q_left * q_right < 0
        # But we need to be careful at y=0
        if y_r < 1e-10:
            # At y=0, check if q changes sign as y increases from 0
            # q(0) = q_coeffs[0], q(eps) should have opposite sign if sign change
            q_at_0 = q_coeffs[0]
            q_at_eps = q_eval(eps)
            if q_at_0 * q_at_eps < 0:
                sign_change_roots.append(y_r)
        else:
            if q_left * q_right < 0:
                sign_change_roots.append(y_r)
    
    if not sign_change_roots:
        return float('inf')
    
    # r is the largest x such that q(x) changes sign
    # y_r = x_r^2, so x_r = sqrt(y_r)
    # r = max(x_r) = sqrt(max(y_r))
    max_y = max(sign_change_roots)
    r = np.sqrt(max_y)
    
    # Check if g tends to negative limit at infinity
    # g(x) ~ c3 * H12(x) ~ c3 * 2^12 * x^12 as x->inf
    # If c3 < 0, g -> -inf, so we replace g by -g
    # This flips the sign of q, which doesn't change the roots but may affect sign changes
    # Actually, replacing g by -g flips q, so sign changes become sign changes still (just flipped)
    # The roots are the same, and sign change roots are the same
    # So this doesn't affect the value
    
    value = r ** 2 / (2.0 * np.pi)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_coeffs = seed.copy()
    best_value = _compute_value(*seed)
    
    # Local search with random perturbations
    np.random.seed(42)
    step_size = 0.1
    
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        # Generate perturbation
        noise = np.random.normal(0, step_size, 3)
        candidate = best_coeffs + noise
        
        # Evaluate
        val = _compute_value(*candidate)
        
        if val < best_value:
            best_value = val
            best_coeffs = candidate.copy()
            # Reduce step size
            step_size *= 0.95
        else:
            # Increase step size slightly if stuck
            step_size *= 1.02
            if step_size > 1.0:
                step_size = 1.0
    
    # Try a few more with smaller steps
    step_size = 0.001
    for iteration in range(2000):
        if time.time() > deadline - 5.0:
            break
        
        noise = np.random.normal(0, step_size, 3)
        candidate = best_coeffs + noise
        val = _compute_value(*candidate)
        
        if val < best_value:
            best_value = val
            best_coeffs = candidate.copy()
    
    return best_coeffs
# EVOLVE-BLOCK-END
