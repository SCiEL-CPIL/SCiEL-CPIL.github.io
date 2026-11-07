"""Optimization: minimize largest sign-changing root of q(x)=g(x)/x^2.
The problem does not require g >= 0; if g -> -inf at infinity, g is replaced by -g,
which doesn't change sign-changing roots. So we just minimize the largest positive
sign-changing root of q over all (c0,c1,c2) where such a root exists."""
import numpy as np
import time
import sys

# Precompute Hermite polynomial coefficients
def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) as a list
    from highest degree to constant term (for numpy.polyval)."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [2.0, 0.0]
    # Build up using recurrence: H_{n+1} = 2x*H_n - 2n*H_{n-1}
    # Represent as coefficient arrays (highest power first)
    h_prev = [1.0]  # H_0
    h_curr = [2.0, 0.0]  # H_1
    for k in range(1, n):
        # h_next = 2x * h_curr - 2k * h_prev
        # 2x * h_curr: shift right by 1, multiply by 2
        shifted = [2.0 * c for c in h_curr] + [0.0]
        # -2k * h_prev: pad to same length
        diff = len(shifted) - len(h_prev)
        padded = [0.0] * diff + [-2.0 * k * c for c in h_prev]
        h_next = [a + b for a, b in zip(shifted, padded)]
        h_prev, h_curr = h_curr, h_next
    return h_curr

# Precompute the coefficient vectors for H0, H4, H8, H12
_H0 = np.array(_hermite_coeffs(0))
_H4 = np.array(_hermite_coeffs(4))
_H8 = np.array(_hermite_coeffs(8))
_H12 = np.array(_hermite_coeffs(12))

# Pad to same length (13 coefficients for degree 12)
_MAX_LEN = 13
def _pad(arr):
    if len(arr) >= _MAX_LEN:
        return arr[:_MAX_LEN]
    return np.concatenate([arr, np.zeros(_MAX_LEN - len(arr))])

_P0 = _pad(_H0)
_P4 = _pad(_H4)
_P8 = _pad(_H8)
_P12 = _pad(_H12)

# Constants for c3 computation
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1680.0
_H12_0 = -1330560.0

def _build_g_coefficients(c0, c1, c2):
    """Build the full polynomial g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    and return coefficients from highest degree to constant, plus c3."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    g_coeffs = c0 * _P0 + c1 * _P4 + c2 * _P8 + c3 * _P12
    
    return g_coeffs, c3

def _q_coefficients(c0, c1, c2):
    """Get coefficients of q(x) = g(x)/x^2 (degree 10 polynomial).
    Returns coefficients from highest degree to constant."""
    g_coeffs, c3 = _build_g_coefficients(c0, c1, c2)
    # g is degree 12, g(x)/x^2 means we drop the last 2 coefficients
    # g_coeffs[0] is x^12, g_coeffs[11] is x^1, g_coeffs[12] is x^0
    # q(x) = g(x)/x^2 has coefficients g_coeffs[0..10]
    q_coeffs = g_coeffs[:11]  # degree 10 down to degree 0
    return q_coeffs

def _find_largest_sign_changing_root(q_coeffs):
    """Find the largest positive real root of the polynomial (coeffs highest first)
    at which the polynomial changes sign (odd multiplicity), using bisection
    on a fine grid for robustness."""
    try:
        # First, use np.roots to get candidate roots
        roots = np.roots(q_coeffs)
        
        # Also do a grid-based sign change detection for robustness
        # Evaluate q on a fine grid to find sign changes
        x_grid = np.linspace(0.01, 100.0, 10000)
        q_vals = np.polyval(q_coeffs, x_grid)
        
        # Find intervals where sign changes
        sign_changes = []
        for i in range(len(q_vals) - 1):
            if q_vals[i] * q_vals[i+1] < 0:
                sign_changes.append((x_grid[i], x_grid[i+1]))
        
        best_r = None
        
        # Refine each sign change interval using bisection
        for lo, hi in sign_changes:
            # Bisection to find the root
            for _ in range(60):
                mid = (lo + hi) / 2.0
                q_mid = np.polyval(q_coeffs, mid)
                q_lo = np.polyval(q_coeffs, lo)
                if q_lo * q_mid <= 0:
                    hi = mid
                else:
                    lo = mid
            r = (lo + hi) / 2.0
            if r > 0 and (best_r is None or r > best_r):
                best_r = r
        
        # Also check np.roots for any larger roots we might have missed
        for r in roots:
            if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real > 0:
                # Verify it's a sign-changing root
                eps = 1e-7 * max(1.0, abs(r.real))
                vl = np.polyval(q_coeffs, r.real - eps)
                vr = np.polyval(q_coeffs, r.real + eps)
                if vl * vr < 0:
                    if best_r is None or r.real > best_r:
                        best_r = r.real
        
        return best_r
    except Exception:
        return None

def _compute_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for the given coefficients.
    Returns value or None if invalid (no sign-changing positive root exists)."""
    q_coeffs = _q_coefficients(c0, c1, c2)
    
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None or r <= 0:
        return None
    
    value = r**2 / (2.0 * np.pi)
    return value

def _objective(params):
    """Objective function for optimization: minimize r^2 (the value).
    Returns a large penalty if no valid sign-changing root exists."""
    c0, c1, c2 = params
    q_coeffs = _q_coefficients(c0, c1, c2)
    
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None or r <= 0:
        return 1e10
    
    value = r**2 / (2.0 * np.pi)
    return value

def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin before 210s limit
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0], dtype=np.float64)
    best_coeffs = seed.copy()
    best_value = _compute_value(*seed)
    if best_value is None:
        best_value = float('inf')
    
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starting_points = [seed.copy()]
    for i in range(20):
        if time.time() > deadline - 10:
            break
        pert = rng.normal(0, 0.1, size=3)
        starting_points.append(seed + pert)
    
    # Run Nelder-Mead from multiple starts
    for x0 in starting_points:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 800,
                    'xatol': 1e-9,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            candidate = result.x
            val = _compute_value(*candidate)
            if val is not None and val < best_value:
                best_value = val
                best_coeffs = candidate.copy()
        except Exception:
            pass
    
    # Intensive local refinement
    if time.time() < deadline - 5:
        current = best_coeffs.copy()
        current_value = best_value
        
        # Multi-scale random local search
        for scale in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 5:
                break
            n_iter = 500
            for _ in range(n_iter):
                if time.time() > deadline - 5:
                    break
                delta = rng.normal(0, scale, size=3)
                new_coeffs = current + delta
                val = _compute_value(*new_coeffs)
                if val is not None and val < current_value:
                    current = new_coeffs
                    current_value = val
                    if current_value < best_value:
                        best_value = current_value
                        best_coeffs = current.copy()
        
        # Directional line search for fine-tuning
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
            np.array([1, 1, 0])/np.sqrt(2), np.array([1, -1, 0])/np.sqrt(2),
            np.array([-1, 1, 0])/np.sqrt(2), np.array([1, 0, 1])/np.sqrt(2),
            np.array([1, 0, -1])/np.sqrt(2), np.array([0, 1, 1])/np.sqrt(2),
            np.array([0, 1, -1])/np.sqrt(2), np.array([1, 1, 1])/np.sqrt(3),
            np.array([1, 1, -1])/np.sqrt(3), np.array([1, -1, 1])/np.sqrt(3),
            np.array([1, -1, -1])/np.sqrt(3), np.array([-1, -1, 1])/np.sqrt(3),
            np.array([-1, -1, -1])/np.sqrt(3),
        ]
        
        for direction in directions:
            if time.time() > deadline - 5:
                break
            for scale in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                if time.time() > deadline - 5:
                    break
                for step in np.linspace(0.01, 1.0, 20):
                    if time.time() > deadline - 5:
                        break
                    new_coeffs = current + scale * step * direction
                    val = _compute_value(*new_coeffs)
                    if val is not None and val < current_value:
                        current = new_coeffs
                        current_value = val
                        if current_value < best_value:
                            best_value = current_value
                            best_coeffs = current.copy()
    
    # Final ultra-fine polish
    if time.time() < deadline - 3:
        for scale in [5e-5, 2e-5, 1e-5, 5e-6]:
            if time.time() > deadline - 3:
                break
            for _ in range(300):
                if time.time() > deadline - 3:
                    break
                delta = rng.normal(0, scale, size=3)
                new_coeffs = best_coeffs + delta
                val = _compute_value(*new_coeffs)
                if val is not None and val < best_value:
                    best_value = val
                    best_coeffs = new_coeffs.copy()
    
    return best_coeffs
