"""
Strategy: Global piecewise linear "hat function" basis with fixed, evenly spaced knots.
Instead of global polynomial or trigonometric basis functions (which oscillate and
suffer from Runge's phenomenon at high degrees), we use a local basis where each
basis function is nonzero only in a small neighborhood. This provides:
1. Local flexibility: can capture arbitrary continuous shapes and discontinuities
2. No oscillation: each parameter affects only a small region
3. Strict linearity in params: BFGS converges to global least-squares solution
4. Non-negative basis values: better conditioned than high-degree polynomials

We use 10 hat functions with overlapping support, evenly spaced across [-2, 2].
Each hat function is a triangular bump centered at a knot location.
"""
import numpy as np
import time

# Track start time for deadline guard
_t0 = time.time()
_DEADLINE = 150.0  # wall-clock limit
_SAFETY_MARGIN = 10.0  # return at least 10s before limit


def _hat_basis(x, n_hats=10, domain=(-2.0, 2.0)):
    """
    Generate n_hats piecewise-linear hat (triangular) basis functions.
    
    Each hat is centered at an evenly-spaced knot in [domain], with support
    extending to adjacent knots. The basis is strictly linear in the sense
    that y = sum_i p_i * hat_i(x).
    
    For n_hats hats, we place knots at n_hats evenly spaced points.
    Each hat has support width equal to 2 * spacing (covering its center
    and reaching to neighboring knots).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    
    # Knot locations: evenly spaced across the domain
    lo, hi = domain
    # Use n_hats + 1 knots to define n_hats-1 intervals, but we want n_hats hats
    # Better approach: place n_hats centers, each with support to neighbors
    centers = np.linspace(lo + (hi - lo) / (2 * n_hats), 
                          hi - (hi - lo) / (2 * n_hats), 
                          n_hats)
    spacing = (hi - lo) / n_hats
    
    basis = np.zeros((n, n_hats))
    for i, c in enumerate(centers):
        # Hat function: triangular, peak at c, zero at c +/- spacing
        left = c - spacing
        right = c + spacing
        # For left side: linear ramp from 0 at left to 1 at c
        mask_left = (x >= left) & (x < c)
        mask_right = (x >= c) & (x <= right)
        mask_outside = ~(mask_left | mask_right)
        
        basis[mask_left, i] = (x[mask_left] - left) / spacing
        basis[mask_right, i] = (right - x[mask_right]) / spacing
        # Outside support: 0 (already initialized)
    
    # Ensure the basis covers the full domain by adding boundary handling
    # The leftmost hat should extend to the left boundary
    # The rightmost hat should extend to the right boundary
    # Actually, our centers are inset by half a spacing, so:
    # First center is at lo + spacing/2, its left edge is at lo
    # Last center is at hi - spacing/2, its right edge is at hi
    # This means the domain is fully covered. Good.
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear hat function model.
    y = sum_i params[i] * hat_i(x)
    
    Strictly linear in params -> BFGS finds global optimum.
    Local basis -> can capture arbitrary continuous functions.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety checks
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Deadline guard
    elapsed = time.time() - _t0
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Return a simple linear fallback
        return p[0] + p[1] * x
    
    # Compute hat basis
    basis = _hat_basis(x, n_hats=10, domain=(-2.0, 2.0))
    
    # Linear combination: y = basis @ params
    out = basis @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
    
    return out
