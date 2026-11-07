"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _get_hermite_coeffs(n):
    """Get coefficients of physicist's Hermite polynomial Hn (highest power first)."""
    # Use recurrence: H0 = 1, H1 = 2x, Hn = 2x*H_{n-1} - 2(n-1)*H_{n-2}
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([2.0, 0.0])
    h_prev2 = np.array([1.0])  # H0
    h_prev1 = np.array([2.0, 0.0])  # H1
    for k in range(2, n + 1):
        # Hk = 2x * H_{k-1} - 2(k-1) * H_{k-2}
        # 2x * H_{k-1}: shift right by 1 and multiply by 2
        term1 = np.zeros(len(h_prev1) + 1)
        term1[:-1] = 2.0 * h_prev1
        # 2(k-1) * H_{k-2}
        term2 = 2.0 * (k - 1) * h_prev2
        # Pad term2 to match length
        if len(term2) < len(term1):
            term2 = np.pad(term2, (0, len(term1) - len(term2)))
        elif len(term2) > len(term1):
            term1 = np.pad(term1, (0, len(term2) - len(term1)))
        h_curr = term1 - term2
        h_prev2, h_prev1 = h_prev1, h_curr
    return h_prev1

def _poly_divide_by_x2(coeffs):
    """Divide polynomial by x^2. coeffs is highest power first.
    Returns None if not divisible, else the quotient coefficients."""
    if len(coeffs) < 3:
        return None
    # Check that last two coefficients are ~0
    if abs(coeffs[-1]) > 1e-10 or abs(coeffs[-2]) > 1e-10:
        return None
    return coeffs[:-2]

def _find_sign_changing_roots(coeffs):
    """Find real roots where the polynomial changes sign."""
    if coeffs is None or len(coeffs) == 0:
        return []
    # Remove leading zeros
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-14:
        coeffs = coeffs[1:]
    if len(coeffs) == 0:
        return []
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    # Check sign change at each real root
    sign_changing = []
    for r in real_roots:
        # Evaluate just before and after
        eps = 1e-8 * (1 + abs(r))
        val_before = np.polyval(coeffs, r - eps)
        val_after = np.polyval(coeffs, r + eps)
        if val_before * val_after < 0:
            sign_changing.append(r)
    return sign_changing

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients."""
    H0 = _get_hermite_coeffs(0)
    H4 = _get_hermite_coeffs(4)
    H8 = _get_hermite_coeffs(8)
    H12 = _get_hermite_coeffs(12)
    
    # Build g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # Pad all to same length
    max_len = max(len(H0), len(H4), len(H8), len(H12))
    def pad(c):
        return np.pad(c, (0, max_len - len(c)))
    
    H0p = pad(H0)
    H4p = pad(H4)
    H8p = pad(H8)
    H12p = pad(H12)
    
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    # Hn(0) is the constant term (last element)
    H0_0 = H0p[-1]
    H4_0 = H4p[-1]
    H8_0 = H8p[-1]
    H12_0 = H12p[-1]
    
    if abs(H12_0) < 1e-14:
        return float('inf')
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = c0 * H0p + c1 * H4p + c2 * H8p + c3 * H12p
    
    # q = g / x^2
    q = _poly_divide_by_x2(g)
    if q is None:
        return float('inf')
    
    # Find largest sign-changing root
    roots = _find_sign_changing_roots(q)
    if len(roots) == 0:
        return float('inf')
    
    r = max(roots)
    return r**2 / (2 * np.pi)

def _smooth_objective(coeffs, temperature=1.0):
    """Smooth surrogate using log-sum-exp over sign-changing roots."""
    c0, c1, c2 = coeffs
    H0 = _get_hermite_coeffs(0)
    H4 = _get_hermite_coeffs(4)
    H8 = _get_hermite_coeffs(8)
    H12 = _get_hermite_coeffs(12)
    
    max_len = max(len(H0), len(H4), len(H8), len(H12))
    def pad(c):
        return np.pad(c, (0, max_len - len(c)))
    
    H0p = pad(H0)
    H4p = pad(H4)
    H8p = pad(H8)
    H12p = pad(H12)
    
    H0_0 = H0p[-1]
    H4_0 = H4p[-1]
    H8_0 = H8p[-1]
    H12_0 = H12p[-1]
    
    if abs(H12_0) < 1e-14:
        return float('inf')
    
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    g = c0 * H0p + c1 * H4p + c2 * H8p + c3 * H12p
    q = _poly_divide_by_x2(g)
    if q is None:
        return float('inf')
    
    roots = _find_sign_changing_roots(q)
    if len(roots) == 0:
        return float('inf')
    
    # Log-sum-exp for soft max
    r_squared = np.array(roots)**2
    max_r2 = np.max(r_squared)
    if max_r2 > 1e10:
        return float('inf')
    log_sum = max_r2 + np.log(np.sum(np.exp(r_squared - max_r2)))
    return log_sum / (2 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s kill
    
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_value = _compute_value(*seed)
    best_coeffs = seed.copy()
    
    # Phase 1: Coarse search with differential evolution
    elapsed = time.time() - start_time
    if elapsed < 100:
        try:
            bounds = [(-5, 5), (-2, 2), (-0.1, 0.1)]
            result = differential_evolution(
                _smooth_objective, bounds, maxiter=50, tol=1e-6,
                popsize=15, seed=42, polish=False, updating='immediate'
            )
            elapsed = time.time() - start_time
            if result.fun < best_value:
                test_val = _compute_value(*result.x)
                if test_val < best_value:
                    best_value = test_val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < 160:
        try:
            result = minimize(
                _smooth_objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            elapsed = time.time() - start_time
            if result.fun < best_value:
                test_val = _compute_value(*result.x)
                if test_val < best_value:
                    best_value = test_val
                    best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with exact objective
    elapsed = time.time() - start_time
    if elapsed < 190:
        try:
            result = minimize(
                _compute_value, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
