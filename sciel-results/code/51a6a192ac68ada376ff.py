"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Precompute Hermite polynomial coefficients (physicists' convention)
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
# H12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480

# Coefficients for H0, H4, H8, H12 (as arrays from x^12 down to x^0)
H0_coeffs = np.array([1.0])
H4_coeffs = np.array([16.0, -48.0, 12.0])  # x^4, x^2, x^0
H8_coeffs = np.array([256.0, -1344.0, 2688.0, -1344.0, 168.0])  # x^8, x^6, x^4, x^2, x^0
H12_coeffs = np.array([4096.0, -46080.0, 332160.0, -1003520.0, 1424640.0, -921600.0, 18480.0])  # x^12, x^10, ..., x^0

# Values at x=0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 168.0
H12_0 = 18480.0


def compute_q_coefficients(c0, c1, c2):
    """Compute the coefficients of q(x) = g(x)/x^2 where g = c0*H0 + c1*H4 + c2*H8 + c3*H12."""
    # c3 is determined by g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build g(x) coefficients (from x^12 down to x^0)
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # We need to align these polynomials properly
    
    # H0: degree 0 -> [1]
    # H4: degree 4 -> [16, 0, -48, 0, 12] (x^4, x^3, x^2, x^1, x^0)
    # H8: degree 8 -> [256, 0, -1344, 0, 2688, 0, -1344, 0, 168]
    # H12: degree 12 -> [4096, 0, -46080, 0, 332160, 0, -1003520, 0, 1424640, 0, -921600, 0, 18480]
    
    # Let's work with full degree-12 polynomial representation
    g_coeffs = np.zeros(13)  # indices 0 to 12, index i = coefficient of x^(12-i)
    
    # H0 contribution: c0 * 1
    g_coeffs[12] += c0 * 1.0
    
    # H4 contribution: c1 * (16x^4 - 48x^2 + 12)
    g_coeffs[8] += c1 * 16.0    # x^4 -> position 12-4=8
    g_coeffs[10] += c1 * (-48.0)  # x^2 -> position 12-2=10
    g_coeffs[12] += c1 * 12.0
    
    # H8 contribution: c2 * (256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168)
    g_coeffs[4] += c2 * 256.0    # x^8 -> position 12-8=4
    g_coeffs[6] += c2 * (-1344.0)  # x^6 -> position 12-6=6
    g_coeffs[8] += c2 * 2688.0   # x^4 -> position 12-4=8
    g_coeffs[10] += c2 * (-1344.0)  # x^2 -> position 12-2=10
    g_coeffs[12] += c2 * 168.0
    
    # H12 contribution: c3 * (4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480)
    g_coeffs[0] += c3 * 4096.0    # x^12
    g_coeffs[2] += c3 * (-46080.0)  # x^10
    g_coeffs[4] += c3 * 332160.0   # x^8
    g_coeffs[6] += c3 * (-1003520.0)  # x^6
    g_coeffs[8] += c3 * 1424640.0  # x^4
    g_coeffs[10] += c3 * (-921600.0)  # x^2
    g_coeffs[12] += c3 * 18480.0
    
    # Check that g(0) = 0 (constant term should be ~0)
    # g_coeffs[12] should be 0
    
    # q(x) = g(x) / x^2
    # Since g(0) = 0, g has a factor of x^2 (actually x^2 since all odd powers are 0)
    # q(x) is degree 10, with coefficients:
    # q_coeffs[i] = g_coeffs[i] for i = 0 to 10 (these are x^10, x^9, ..., x^0 in q)
    # But wait, we need to be careful about the indexing
    
    # g(x) = sum_{k=0}^{12} g_coeffs[k] * x^(12-k)
    # g(x) = x^2 * q(x) where q(x) = sum_{j=0}^{10} q_coeffs[j] * x^(10-j)
    # So g_coeffs[k] = q_coeffs[k] for k = 0 to 10
    # And g_coeffs[11] = 0 (x^1 term, which should be 0 since g is even)
    # And g_coeffs[12] = 0 (x^0 term, which should be 0 since g(0)=0)
    
    q_coeffs = g_coeffs[:11]  # Coefficients of q from x^10 down to x^0
    
    return q_coeffs


def evaluate_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) where r is the largest real root of q where q changes sign."""
    q_coeffs = compute_q_coefficients(c0, c1, c2)
    
    # Find roots of q(x)
    try:
        roots = np.roots(q_coeffs)
    except Exception:
        return float('inf')
    
    # Filter for real roots
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            real_roots.append(root.real)
    
    if not real_roots:
        return float('inf')
    
    # For each real root, check if q changes sign at that root
    # q changes sign at a root if the multiplicity is odd
    # We can check by evaluating q at points slightly before and after the root
    
    sign_change_roots = []
    for r in real_roots:
        if r <= 0:
            continue
        
        # Evaluate q at r - eps and r + eps
        eps = 1e-8
        q_before = np.polyval(q_coeffs, r - eps)
        q_after = np.polyval(q_coeffs, r + eps)
        
        # Check for sign change
        if q_before * q_after < 0:
            sign_change_roots.append(r)
    
    if not sign_change_roots:
        return float('inf')
    
    r = max(sign_change_roots)
    value = r**2 / (2 * np.pi)
    
    return value


def construct():
    """Search for optimal coefficients c0, c1, c2."""
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit
    
    # Initial best from seed
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_value = evaluate_value(best_c[0], best_c[1], best_c[2])
    
    # Local optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(coeffs):
        return evaluate_value(coeffs[0], coeffs[1], coeffs[2])
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            best_c,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if result.fun < best_value and time.time() < deadline:
            best_c = result.x
            best_value = result.fun
    except Exception:
        pass
    
    # Try a few random perturbations around the best found
    np.random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Random perturbation
        scale = 0.1
        perturbation = np.random.randn(3) * scale
        candidate = best_c + perturbation
        
        val = objective(candidate)
        if val < best_value:
            best_c = candidate
            best_value = val
    
    # Final local refinement
    if time.time() < deadline - 30:
        try:
            result = minimize(
                objective,
                best_c,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.fun < best_value:
                best_c = result.x
                best_value = result.fun
        except Exception:
            pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
