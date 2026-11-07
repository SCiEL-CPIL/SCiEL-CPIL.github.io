"""Constrained optimization: minimize largest sign-changing root of q(x)=g(x)/x^2
subject to g(x) >= 0 for all x >= 0, using penalty-based Nelder-Mead."""
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
    """Find the largest real root of the polynomial (coeffs highest first)
    at which the polynomial changes sign (odd multiplicity)."""
    try:
        # Use numpy to find all roots
        roots = np.roots(q_coeffs)
        # Filter to real roots (imaginary part close to 0)
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10 * max(1.0, abs(r.real)):
                real_roots.append(r.real)
        
        if not real_roots:
            return None
        
        # For each real root, check if it's a sign-changing root (odd multiplicity)
        best_r = None
        for r in real_roots:
            if r <= 0:
                continue
            # Check sign change: evaluate at r - eps and r + eps
            eps = 1e-8 * max(1.0, abs(r))
            val_left = np.polyval(q_coeffs, r - eps)
            val_right = np.polyval(q_coeffs, r + eps)
            # Sign change means different signs
            if val_left * val_right < 0:
                if best_r is None or r > best_r:
                    best_r = r
        
        return best_r
    except Exception:
        return None

def _check_nonnegativity(g_coeffs, n_points=500, x_max=40.0):
    """Check that g(x) >= 0 for x >= 0 on a grid. Returns min value."""
    x_grid = np.linspace(0, x_max, n_points)
    g_vals = np.polyval(g_coeffs, x_grid)
    return np.min(g_vals)

def _compute_value(c0, c1, c2, strict=False):
    """Compute the value = r^2 / (2*pi) for the given coefficients.
    Returns value or None if invalid.
    If strict=True, use a finer grid for non-negativity check."""
    q_coeffs = _q_coefficients(c0, c1, c2)
    g_coeffs, _ = _build_g_coefficients(c0, c1, c2)
    
    # Check that g(x) >= 0 for x >= 0
    if strict:
        min_val = _check_nonnegativity(g_coeffs, n_points=2000, x_max=50.0)
    else:
        min_val = _check_nonnegativity(g_coeffs, n_points=500, x_max=40.0)
    
    if min_val < -1e-10:
        return None  # Violates non-negativity
    
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None or r <= 0:
        return None
    
    value = r**2 / (2.0 * np.pi)
    return value

def _objective(params):
    """Objective function for optimization: minimize r^2 subject to g >= 0.
    Uses smooth epigraph formulation with soft-min surrogate for constraint."""
    c0, c1, c2 = params
    q_coeffs = _q_coefficients(c0, c1, c2)
    g_coeffs, _ = _build_g_coefficients(c0, c1, c2)
    
    # Check non-negativity with smooth soft-min penalty
    x_grid = np.linspace(0, 40.0, 500)
    g_vals = np.polyval(g_coeffs, x_grid)
    # Soft-min: smooth approximation of min(g_vals) using log-sum-exp
    # min(g) ≈ -log-sum-exp(-g) / n, but we want a penalty for min(g) < 0
    # Use: penalty = softplus(-min(g)) where softplus(z) = log(1+exp(z))
    # For numerical stability, use: softplus(z) ≈ z for z > 30, 0 for z < -30
    min_g = np.min(g_vals)
    # Smooth penalty: use log(1 + exp(-k*min_g)) / k as smooth max(0, -min_g)
    k = 10.0
    if min_g < -30.0 / k:
        penalty = -min_g  # Far below zero, linear regime
    else:
        penalty = np.log(1.0 + np.exp(-k * min_g)) / k
    
    # Find largest sign-changing root
    r = _find_largest_sign_changing_root(q_coeffs)
    if r is None or r <= 0:
        return 1e10 + penalty * 100.0
    
    value = r**2 / (2.0 * np.pi)
    # Smooth penalty for constraint violation
    return value + penalty * 100.0

def construct():
    """Return three coefficients [c0, c1, c2] that minimize the value."""
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 210s limit
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0], dtype=np.float64)
    best_coeffs = seed.copy()
    best_value = _compute_value(*seed, strict=True)
    if best_value is None:
        best_value = float('inf')
    
    # Nelder-Mead optimization with penalty
    from scipy.optimize import minimize
    
    # Try multiple starting points
    rng = np.random.RandomState(12345)
    
    starting_points = [seed.copy()]
    # Add some random perturbations
    for i in range(5):
        if time.time() > deadline:
            break
        pert = rng.normal(0, 0.05, size=3)
        starting_points.append(seed + pert)
    
    for x0 in starting_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_value:
                candidate = result.x
                val = _compute_value(*candidate, strict=False)
                if val is not None and val < best_value:
                    best_value = val
                    best_coeffs = candidate.copy()
        except Exception:
            pass
    
    # Local refinement around best point found
    if time.time() < deadline:
        current = best_coeffs.copy()
        current_value = best_value
        
        # Multi-scale local search
        for scale in [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]:
            for _ in range(300):
                if time.time() > deadline:
                    break
                
                delta = rng.normal(0, scale, size=3)
                new_coeffs = current + delta
                
                val = _compute_value(*new_coeffs, strict=False)
                if val is not None and val < current_value:
                    current = new_coeffs
                    current_value = val
                    if current_value < best_value:
                        best_value = current_value
                        best_coeffs = current.copy()
            
            if time.time() > deadline:
                break
        
        # Directional search
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
            np.array([1, 1, 0]), np.array([1, -1, 0]),
            np.array([-1, 1, 0]), np.array([1, 0, 1]),
            np.array([1, 0, -1]), np.array([0, 1, 1]),
            np.array([0, 1, -1]), np.array([1, 1, 1]),
            np.array([1, 1, -1]), np.array([1, -1, 1]),
            np.array([1, -1, -1]), np.array([-1, -1, 1]),
            np.array([-1, -1, -1]),
        ]
        
        for direction in directions:
            if time.time() > deadline:
                break
            direction = direction / np.linalg.norm(direction)
            
            for scale in [0.01, 0.005, 0.001, 0.0005, 0.0002]:
                for step in range(30):
                    if time.time() > deadline:
                        break
                    new_coeffs = current + scale * step * 0.1 * direction
                    val = _compute_value(*new_coeffs, strict=False)
                    if val is not None and val < current_value:
                        current = new_coeffs
                        current_value = val
                        if current_value < best_value:
                            best_value = current_value
                            best_coeffs = current.copy()
    
    # Final polish: very fine local search with strict checking
    if time.time() < deadline - 5:
        for scale in [0.00001, 0.000005, 0.000001]:
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                delta = rng.normal(0, scale, size=3)
                new_coeffs = best_coeffs + delta
                val = _compute_value(*new_coeffs, strict=True)
                if val is not None and val < best_value:
                    best_value = val
                    best_coeffs = new_coeffs.copy()
        
        # High-precision polish using scipy minimize with tight tolerances
        if time.time() < deadline - 3:
            try:
                from scipy.optimize import minimize as scipy_minimize
                # Use BFGS with analytical gradient approximation via finite differences
                # but with very tight tolerances
                result = scipy_minimize(
                    _objective,
                    best_coeffs,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if result.fun < best_value:
                    candidate = result.x
                    val = _compute_value(*candidate, strict=True)
                    if val is not None and val < best_value:
                        best_value = val
                        best_coeffs = candidate.copy()
            except Exception:
                pass
    
    # Verify final result with strict checking
    final_val = _compute_value(*best_coeffs, strict=True)
    if final_val is not None and final_val < best_value:
        best_value = final_val
    
    # One more refinement pass: try small perturbations in each coordinate
    if time.time() < deadline - 1:
        for i in range(3):
            for sign in [-1, 1]:
                for delta in [1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
                    if time.time() > deadline - 1:
                        break
                    new_coeffs = best_coeffs.copy()
                    new_coeffs[i] += sign * delta
                    val = _compute_value(*new_coeffs, strict=True)
                    if val is not None and val < best_value:
                        best_value = val
                        best_coeffs = new_coeffs.copy()
    
    return best_coeffs
