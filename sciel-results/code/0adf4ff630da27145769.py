"""
Uncertainty inequality: tune Hermite-Gaussian test function coefficients to minimize the upper bound on C4.

We parameterize g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is derived from the constraint g(0) = 0.

The score is the upper bound on A(f)*A(Fourier(f)) (or C4-related constant).
We use analytic expressions for norm^2 and fast numerical evaluation for norm_inf.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE_FRACTION = 0.88  # safety margin before 210s limit

# Precompute Hermite polynomials as coefficient arrays
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 60x^10 + 1260x^8 - 10080x^6 + 30240x^4 - 30240x^2 + 945

# Coefficients for H_n: [c_12, c_10, c_8, c_6, c_4, c_2, c_0]
_H_COEFFS = np.array([
    [0, 0, 0, 0, 0, 0, 1],       # H0
    [0, 0, 0, 0, 1, -6, 3],      # H4
    [0, 0, 1, -28, 210, -420, 105],  # H8
    [1, -60, 1260, -10080, 30240, -30240, 945]  # H12
])

# Precompute a smaller grid for norm_inf evaluation
_X_GRID = np.linspace(-10, 10, 20000)
_DX = _X_GRID[1] - _X_GRID[0]
_GAUSS = np.exp(-0.5 * _X_GRID**2)

# Precompute powers of x for fast polynomial evaluation
_X_POWERS = np.column_stack([
    _X_GRID**12, _X_GRID**10, _X_GRID**8, _X_GRID**6,
    _X_GRID**4, _X_GRID**2, np.ones_like(_X_GRID)
])


def _compute_norm2_sq(a0, a4, a8):
    """
    Compute ||g||_2^2 analytically using Hermite orthogonality.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)
    ||g||_2^2 = integral of g(x)^2 dx = integral of (poly)^2 * exp(-x^2) dx
    
    Using orthogonality: integral H_m * H_n * exp(-x^2) dx = 0 if m != n,
    and integral H_n^2 * exp(-x^2) dx = sqrt(pi) * 2^n * n!
    """
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Coefficients: [a0, a4, a8, a12]
    coeffs = np.array([a0, a4, a8, a12])
    
    # Norm squared = sum of c_i^2 * integral H_i^2 * exp(-x^2) dx
    # integral H_n^2 * exp(-x^2) dx = sqrt(pi) * 2^n * n!
    norms_sq = [
        np.sqrt(np.pi) * 2**0 * 1,          # n=0: sqrt(pi) * 1 * 1
        np.sqrt(np.pi) * 2**4 * 24,          # n=4: sqrt(pi) * 16 * 24
        np.sqrt(np.pi) * 2**8 * 40320,       # n=8: sqrt(pi) * 256 * 40320
        np.sqrt(np.pi) * 2**12 * 479001600   # n=12: sqrt(pi) * 4096 * 479001600
    ]
    
    norm2_sq = 0.0
    for i, c in enumerate(coeffs):
        norm2_sq += c**2 * norms_sq[i]
    
    return norm2_sq


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on the uncertainty product for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    """
    # Derive a12 from g(0) = 0
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Build g(x) using precomputed arrays
    poly = a0*_H_COEFFS[0] + a4*_H_COEFFS[1] + a8*_H_COEFFS[2] + a12*_H_COEFFS[3]
    g = _X_POWERS @ poly * _GAUSS
    
    # ||g||_2^2 (analytic)
    norm2_sq = _compute_norm2_sq(a0, a4, a8)
    
    if norm2_sq <= 0:
        return 1e10  # invalid
    
    norm2 = np.sqrt(norm2_sq)
    norm_inf = np.max(np.abs(g))
    
    bound = (norm_inf / norm2)**2
    
    return bound


def construct():
    start_time = time.time()
    deadline = 210.0 * DEADLINE_FRACTION  # ~184.8s
    
    best_val = float('inf')
    best_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])  # seed
    
    # Evaluate seed
    try:
        seed_val = _compute_bound(best_x[0], best_x[1], best_x[2])
        if seed_val < best_val:
            best_val = seed_val
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_x
    
    # Phase 1: Coarse grid search to explore the landscape
    try:
        a0_center, a4_center, a8_center = best_x
        a0_span, a4_span, a8_span = 3.0, 0.3, 0.008
        
        n_a0, n_a4, n_a8 = 15, 12, 10
        
        a0_vals = np.linspace(a0_center - a0_span, a0_center + a0_span, n_a0)
        a4_vals = np.linspace(a4_center - a4_span, a4_center + a4_span, n_a4)
        a8_vals = np.linspace(a8_center - a8_span, a8_center + a8_span, n_a8)
        
        for a0 in a0_vals:
            for a4 in a4_vals:
                for a8 in a8_vals:
                    if time.time() - start_time > deadline - 55:
                        raise TimeoutError
                    try:
                        v = _compute_bound(a0, a4, a8)
                        if v < best_val:
                            best_val = v
                            best_x = np.array([a0, a4, a8])
                    except Exception:
                        pass
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        return best_x
    
    # Phase 2: Nelder-Mead refinement from the best grid point
    try:
        result = minimize(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 25:
        return best_x
    
    # Phase 3: Multi-start refinement from perturbed points around the best solution
    np.random.seed(123)
    try:
        for i in range(10):
            if time.time() - start_time > deadline - 15:
                break
            # Varying perturbation scales
            scale = 0.02 * (1.0 + 0.5*i)
            x0 = best_x + scale * np.random.randn(3)
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Phase 4: Final high-precision polish
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
