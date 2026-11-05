"""
Optimize 8 points to maximize the normalized product of pairwise distances
(discriminant / diameter^(n(n-1))).

Approach:
- Represent points as roots of a monic complex polynomial P(z) = z^8 + c7*z^7 + ... + c0.
- The 8 complex coefficients (16 real params) are optimized.
- Objective: maximize log|discriminant| - n(n-1)*log(diameter),
  where |discriminant| = prod_{i<j} |z_i - z_j|^2.
- Use scipy.optimize with multiple random restarts and local refinement.
"""

import numpy as np
from math import cos, sin, pi, log, sqrt
from scipy.optimize import minimize, differential_evolution

def _distances(points):
    """Return matrix of pairwise distances."""
    n = len(points)
    pts = np.array(points)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)

def _score_from_roots(roots):
    """
    Compute the score from complex roots.
    Score = log(prod_{i!=j} |z_i - z_j|) - n(n-1)*log(diameter)
         = 2*log(prod_{i<j} |z_i - z_j|) - n(n-1)*log(diameter)
    """
    n = 8
    diff = roots[:, np.newaxis] - roots[np.newaxis, :]
    dists = np.abs(diff)
    np.fill_diagonal(dists, 0)
    
    # Check for non-finite values
    if not np.all(np.isfinite(dists)):
        return -np.inf
    
    # Handle zero distances
    dists = np.maximum(dists, 1e-15)
    
    # Product over unordered pairs
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    log_prod_unordered = np.sum(np.log(pair_dists))
    
    # Diameter (max distance)
    diam = np.max(dists)
    if diam < 1e-15:
        return -np.inf
    
    # Score: 2*log_prod_unordered - n(n-1)*log(diam)
    val = 2 * log_prod_unordered - n * (n - 1) * np.log(diam)
    
    # Penalty for points being too close
    min_dist = np.min(pair_dists)
    if min_dist < 1e-6:
        val -= 1e6 * (1e-6 - min_dist)
    
    return val

def _objective_poly(x):
    """
    Objective to minimize: -score
    x is 16-vector: [c0_re, c0_im, c1_re, c1_im, ..., c7_re, c7_im]
    where P(z) = z^8 + c7*z^7 + ... + c1*z + c0
    """
    n = 8
    # Reconstruct complex coefficients
    coeffs = x[0::2] + 1j * x[1::2]
    
    # Build polynomial coefficients for numpy.roots: [1, c7, c6, ..., c0]
    poly_coeffs = np.concatenate([[1.0], coeffs[::-1]])
    
    # Find roots
    roots = np.roots(poly_coeffs)
    
    # Check that roots are finite
    if not np.all(np.isfinite(roots)):
        return 1e10
    
    score = _score_from_roots(roots)
    
    if not np.isfinite(score):
        return 1e10
    
    return -score

def construct():
    """
    Returns 8 distinct points [x, y] that maximize the normalized product
    of pairwise distances.
    
    Uses complex polynomial representation: points are roots of a monic
    degree-8 polynomial. Optimize the 8 complex coefficients (16 real params)
    to maximize log|discriminant| - n(n-1)*log(diameter).
    """
    n = 8
    best_score = -np.inf
    best_points = None
    
    # Reference: regular octagon
    reg_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    reg_roots = reg_pts[:, 0] + 1j * reg_pts[:, 1]
    dist_reg = _distances(reg_pts)
    np.fill_diagonal(dist_reg, 0)
    diam_reg = np.max(dist_reg)
    log_prod_reg = np.sum(np.log(np.maximum(dist_reg[np.triu_indices(n, k=1)], 1e-15)))
    ref_score = 2 * log_prod_reg - n * (n - 1) * np.log(diam_reg)
    best_score = ref_score
    best_points = reg_pts.copy()
    
    # Get initial coefficients from regular octagon
    # P(z) = prod(z - r_k) = z^8 + c7*z^7 + ... + c0
    reg_poly = np.poly(reg_roots)
    # reg_poly[0] = 1, reg_poly[1] = c7, ..., reg_poly[8] = c0
    reg_coeffs = reg_poly[1:][::-1]  # [c0, c1, ..., c7]
    
    def coeffs_to_flat(coeffs):
        """Convert complex coefficients array to flat real vector."""
        flat = np.zeros(16)
        flat[0::2] = np.real(coeffs)
        flat[1::2] = np.imag(coeffs)
        return flat
    
    def flat_to_coeffs(flat):
        """Convert flat real vector to complex coefficients array."""
        return flat[0::2] + 1j * flat[1::2]
    
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if restart == 0:
            # Start from regular octagon coefficients with small perturbation
            x0 = coeffs_to_flat(reg_coeffs) + rng.normal(0, 0.01, 16)
        else:
            # Random complex coefficients
            x0 = rng.normal(0, 1.0, 16)
        
        # Use L-BFGS-B for local optimization
        result = minimize(_objective_poly, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12})
        
        if result.fun < -best_score:
            # Extract points from roots
            coeffs = flat_to_coeffs(result.x)
            poly_coeffs = np.concatenate([[1.0], coeffs[::-1]])
            roots = np.roots(poly_coeffs)
            
            # Check that roots are finite
            if np.all(np.isfinite(roots)):
                pts = np.column_stack([np.real(roots), np.imag(roots)])
                
                # Verify distinctness and compute actual score
                dist = _distances(pts)
                np.fill_diagonal(dist, 0)
                if np.min(dist) > 1e-8:
                    score = _score_from_roots(roots)
                    if score > best_score:
                        best_score = score
                        best_points = pts
    
    # Try a few more refinements with tighter tolerances
    if best_points is not None:
        best_roots = best_points[:, 0] + 1j * best_points[:, 1]
        best_poly = np.poly(best_roots)
        best_coeffs = best_poly[1:][::-1]
        best_x_flat = coeffs_to_flat(best_coeffs)
        
        for _ in range(5):
            result = minimize(_objective_poly, best_x_flat, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-14})
            
            if result.fun < -best_score:
                coeffs = flat_to_coeffs(result.x)
                poly_coeffs = np.concatenate([[1.0], coeffs[::-1]])
                roots = np.roots(poly_coeffs)
                
                if np.all(np.isfinite(roots)):
                    pts = np.column_stack([np.real(roots), np.imag(roots)])
                    dist = _distances(pts)
                    np.fill_diagonal(dist, 0)
                    if np.min(dist) > 1e-8:
                        score = _score_from_roots(roots)
                        if score > best_score:
                            best_score = score
                            best_points = pts
                            # Update x_flat for next iteration
                            best_roots = best_points[:, 0] + 1j * best_points[:, 1]
                            best_poly = np.poly(best_roots)
                            best_coeffs = best_poly[1:][::-1]
                            best_x_flat = coeffs_to_flat(best_coeffs)
    
    # Final check: ensure points are valid
    pts = best_points
    if pts is None or np.any(np.abs(pts) >= 1e6):
        pts = reg_pts
    
    # Verify distinctness
    dist = _distances(pts)
    np.fill_diagonal(dist, 0)
    if np.min(dist) < 1e-8:
        pts = reg_pts
    
    return [[float(x), float(y)] for x, y in pts]
