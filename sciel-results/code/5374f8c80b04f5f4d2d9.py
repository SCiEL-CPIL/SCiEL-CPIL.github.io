"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 210.0
_SAFETY_MARGIN = 5.0  # seconds


def _hermite_coeffs(n):
    """Return coefficients of the physicist's Hermite polynomial H_n(x) as a list [a0, a1, ..., an]."""
    # Use recurrence: H_0 = 1, H_1 = 2x, H_{n+1} = 2x*H_n - 2n*H_{n-1}
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    h_prev = [1.0]  # H_0
    h_curr = [0.0, 2.0]  # H_1
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2*(k-1) * H_{k-2}
        # 2x * H_{k-1}: shift coefficients by 1 and multiply by 2
        term1 = [0.0] + [2.0 * c for c in h_curr]
        # 2*(k-1) * H_{k-2}
        term2 = [2.0 * (k - 1) * c for c in h_prev]
        # H_k = term1 - term2
        max_len = max(len(term1), len(term2))
        term1 = term1 + [0.0] * (max_len - len(term1))
        term2 = term2 + [0.0] * (max_len - len(term2))
        h_new = [t1 - t2 for t1, t2 in zip(term1, term2)]
        h_prev = h_curr
        h_curr = h_new
    return h_curr


def _eval_hermite(coeffs, x):
    """Evaluate Hermite polynomial with given coefficients at x (scalar or array)."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(reversed(coeffs)):
        result = result * x + c
    return result


def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F) using analytic moments."""
    # Degrees involved
    degrees = [0, 4, 8, 12]
    
    # Evaluate H_n(0) for constraint
    # H_0(0) = 1
    # H_4(0) = -12
    # H_8(0) = 1344
    # H_12(0) = -499456
    # Let's verify with _hermite_coeffs just in case, though these are standard.
    h0_0 = _eval_hermite(_hermite_coeffs(0), np.array([0.0]))[0]
    h4_0 = _eval_hermite(_hermite_coeffs(4), np.array([0.0]))[0]
    h8_0 = _eval_hermite(_hermite_coeffs(8), np.array([0.0]))[0]
    h12_0 = _eval_hermite(_hermite_coeffs(12), np.array([0.0]))[0]
    
    if abs(h12_0) < 1e-15:
        return 1e10
        
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    # Compute Norm squared: sum_k c_k^2 * sqrt(pi) * 2^k * k!
    norm_sq = 0.0
    # Compute x2 moment: sum_{k,j} c_k c_j * Integral(x^2 H_k H_j e^{-x^2})
    # Using x^2 H_k = 0.25 * (H_{k+2} + 2k H_k + k(k-1) H_{k-2})
    # Integral(H_m H_n e^{-x^2}) = sqrt(pi) * 2^n * n! * delta_{mn}
    # So Integral(x^2 H_k H_j e^{-x^2}) is non-zero only if the expanded terms match indices.
    
    # Let's precompute the factorials and powers
    max_deg = 14 # max degree in expansion of x^2 H_12 is 14
    fact = [1.0] * (max_deg + 1)
    pow2 = [1.0] * (max_deg + 1)
    for i in range(1, max_deg + 1):
        fact[i] = fact[i-1] * i
        pow2[i] = pow2[i-1] * 2
    
    # Norm
    for i, k in enumerate(degrees):
        norm_sq += coeffs[i]**2 * np.sqrt(np.pi) * pow2[k] * fact[k]
        
    if norm_sq < 1e-15:
        return 1e10
        
    x2_moment = 0.0
    for i, k in enumerate(degrees):
        for j, l in enumerate(degrees):
            # We need <H_l, x^2 H_k> = <x^2 H_k, H_l>
            # x^2 H_k = 0.25 H_{k+2} + 0.5 k H_k + 0.25 k(k-1) H_{k-2}
            # Term 1: 0.25 <H_l, H_{k+2}> -> non-zero if l == k+2
            # Term 2: 0.5 k <H_l, H_k> -> non-zero if l == k
            # Term 3: 0.25 k(k-1) <H_l, H_{k-2}> -> non-zero if l == k-2 (and k>=2)
            
            val = 0.0
            # Check if l == k+2
            if l == k + 2:
                val += 0.25 * np.sqrt(np.pi) * pow2[l] * fact[l]
            # Check if l == k
            elif l == k:
                val += 0.5 * k * np.sqrt(np.pi) * pow2[l] * fact[l]
            # Check if l == k-2
            elif l == k - 2:
                if k >= 2:
                    val += 0.25 * k * (k-1) * np.sqrt(np.pi) * pow2[l] * fact[l]
            
            x2_moment += coeffs[i] * coeffs[j] * val
            
    A_x = np.sqrt(x2_moment / norm_sq)
    product = A_x * A_x
    return product


def construct():
    # Try to optimize within the time budget
    best_score = _compute_bound(-113/100, 1/25, 1/3240)
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return best_coeffs
    
    # Try some local optimization
    try:
        x0 = np.array([-113/100, 1/25, 1/3240])
        
        def objective(coeffs):
            if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                return 1e10
            c0, c4, c8 = coeffs
            return _compute_bound(c0, c4, c8)
        
        # Try Nelder-Mead
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
        
        # Try a few more starting points if time allows
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - _SAFETY_MARGIN - 10:
            # Try perturbed starting points
            for trial in range(3):
                if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
                    break
                x0_trial = x0 + np.random.normal(0, 0.1, 3)
                result = minimize(objective, x0_trial, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10,
                                          'adaptive': True})
                if result.fun < best_score:
                    best_score = result.fun
                    best_coeffs = result.x
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
