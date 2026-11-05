"""
Optimize 8 points to maximize the normalized product of pairwise distances
using a complex polynomial discriminant formulation.

Approach:
- Represent 8 points as roots of a monic degree-8 complex polynomial.
- Optimize the 16 real parameters (8 complex coefficients) to maximize
  log|discriminant| - n(n-1)*log(diameter).
- Use numpy.roots to extract points from coefficients at each evaluation.
- Multiple random restarts with local refinement.
"""

import numpy as np
from math import cos, sin, pi, log, sqrt
from scipy.optimize import minimize

def _compute_objective_from_points(pts):
    """Compute the objective value from 8x2 array of points."""
    n = 8
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    log_prod_unordered = np.sum(np.log(pair_dists))
    
    diam = np.max(dist)
    
    # Objective: sum over ordered pairs of log(dist) - n(n-1)*log(diam)
    # = 2 * sum over unordered pairs of log(dist) - n(n-1)*log(diam)
    val = 2 * log_prod_unordered - n * (n - 1) * np.log(diam)
    
    # Penalty for points being too close
    min_dist = np.min(pair_dists)
    if min_dist < 1e-6:
        val -= 1e6 * (1e-6 - min_dist)
    
    # Penalty for points going out of bounds
    if np.any(np.abs(pts) >= 1e6):
        val -= 1e10
    
    return val

def _coeffs_to_points(coeffs):
    """Convert 8 complex coefficients (excluding leading 1) to 8 points.
    coeffs is a complex array of length 8: [c7, c6, ..., c0] for z^7 + c7*z^6 + ... + c0
    Actually, numpy.roots expects [a7, a6, ..., a0] where a7 is leading.
    For monic polynomial: [1, c1, c2, ..., c8]
    """
    # coeffs should be the 8 sub-leading coefficients
    poly_coeffs = np.concatenate([[1.0 + 0j], coeffs])
    try:
        roots = np.roots(poly_coeffs)
    except:
        return None
    return np.column_stack([roots.real, roots.imag])

def _points_to_coeffs(pts):
    """Convert 8 points to the 8 sub-leading coefficients of the monic polynomial."""
    # Build polynomial from roots
    roots = pts[:, 0] + 1j * pts[:, 1]
    # np.poly(roots) returns [1, c1, c2, ..., c8]
    poly_coeffs = np.poly(roots)
    return poly_coeffs[1:]

def _objective_coeff(x):
    """Objective to minimize: -val where val is computed from points derived from coefficients.
    x is 16-vector: [Re(c1), Im(c1), Re(c2), Im(c2), ..., Re(c8), Im(c8)]
    """
    coeffs = x[0::2] + 1j * x[1::2]
    pts = _coeffs_to_points(coeffs)
    
    if pts is None:
        return 1e10
    
    # Check for NaN or Inf
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    val = _compute_objective_from_points(pts)
    return -val

def construct():
    """
    Returns 8 distinct points [x, y] that maximize the normalized product
    of pairwise distances.
    
    Uses polynomial discriminant formulation with numerical optimization.
    """
    n = 8
    best_score = -np.inf
    best_points = None
    
    # Reference: regular octagon
    reg_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    ref_score = _compute_objective_from_points(reg_pts)
    best_score = ref_score
    best_points = reg_pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Start from regular octagon coefficients
    reg_coeffs = _points_to_coeffs(reg_pts)
    reg_x = np.zeros(16)
    reg_x[0::2] = reg_coeffs.real
    reg_x[1::2] = reg_coeffs.imag
    
    for restart in range(15):
        if restart == 0:
            x0 = reg_x + rng.normal(0, 0.01, 16)
        elif restart == 1:
            x0 = reg_x + rng.normal(0, 0.05, 16)
        elif restart == 2:
            x0 = reg_x + rng.normal(0, 0.1, 16)
        else:
            # Random points in unit disk
            angles = rng.uniform(0, 2 * pi, 8)
            radii = rng.uniform(0.3, 1.0, 8)
            pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            coeffs_init = _points_to_coeffs(pts_init)
            x0 = np.zeros(16)
            x0[0::2] = coeffs_init.real
            x0[1::2] = coeffs_init.imag
        
        result = minimize(_objective_coeff, x0, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-12})
        
        if result.fun < -best_score:
            coeffs_opt = result.x[0::2] + 1j * result.x[1::2]
            pts_opt = _coeffs_to_points(coeffs_opt)
            if pts_opt is not None and np.all(np.isfinite(pts_opt)):
                score = _compute_objective_from_points(pts_opt)
                if score > best_score:
                    best_score = score
                    best_points = pts_opt
    
    # Refine best solution
    if best_points is not None:
        best_coeffs = _points_to_coeffs(best_points)
        best_x = np.zeros(16)
        best_x[0::2] = best_coeffs.real
        best_x[1::2] = best_coeffs.imag
        
        for _ in range(3):
            result = minimize(_objective_coeff, best_x, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-14})
            if result.fun < -best_score:
                coeffs_opt = result.x[0::2] + 1j * result.x[1::2]
                pts_opt = _coeffs_to_points(coeffs_opt)
                if pts_opt is not None and np.all(np.isfinite(pts_opt)):
                    score = _compute_objective_from_points(pts_opt)
                    if score > best_score:
                        best_score = score
                        best_points = pts_opt
                        best_coeffs = _points_to_coeffs(best_points)
                        best_x[0::2] = best_coeffs.real
                        best_x[1::2] = best_coeffs.imag
    
    # Final validation
    pts = best_points
    if pts is None or np.any(np.abs(pts) >= 1e6):
        pts = reg_pts
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 0)
    if np.min(dist) < 1e-8:
        pts = reg_pts
    
    return [[float(x), float(y)] for x, y in pts]
