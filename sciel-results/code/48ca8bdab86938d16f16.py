"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def _compute_score(centres):
    """Compute the packing score: min(min pairwise dist / 2, min boundary dist)."""
    n = len(centres)
    if n < 2:
        return 1.0 - np.max(np.linalg.norm(centres, axis=1))
    dists = pdist(centres)
    min_pair = np.min(dists) / 2.0
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary = np.min(boundary_dists)
    return min(min_pair, min_boundary)


def _hexagonal_initial(n, max_radius=0.95):
    """Generate initial hexagonal lattice points inside a disk."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= max_radius:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= n:
            # Sort by distance from center to get most central points
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists, kind="stable")[:n]
            return points[idx]
        spacing *= 0.97


def _make_residuals(n, r_target):
    """Create a residuals function for least_squares with target radius r_target."""
    def residuals(x):
        centres = x[:2*n].reshape(n, 2)
        
        # Pairwise distance residuals: want d_ij >= 2*r_target
        # residual = max(0, 2*r_target - d_ij)
        dists = pdist(centres)
        pair_res = np.maximum(0, 2.0 * r_target - dists)
        
        # Boundary residuals: want 1 - |c_i| >= r_target
        # residual = max(0, r_target - (1 - |c_i|))
        boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
        boundary_res = np.maximum(0, r_target - boundary_dists)
        
        return np.concatenate([pair_res, boundary_res])
    
    return residuals


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses progressive least-squares optimization with explicit target radius.
    """
    n = N
    
    # Start with hexagonal lattice initialization
    centres = _hexagonal_initial(n, 0.95)
    current_score = _compute_score(centres)
    
    # Progressive optimization: gradually increase target radius
    # Start from current score and try to push higher
    r_start = current_score * 0.95  # Start slightly below current to ensure feasibility
    r_end = 0.0532  # Best known packing radius
    
    # Create a sequence of target radii to try
    # Use logarithmic spacing for better coverage
    num_stages = 8
    r_targets = np.geomspace(r_start, r_end, num_stages)
    
    best_centres = centres.copy()
    best_score = current_score
    
    for r_target in r_targets:
        # Skip if target is not higher than current best
        if r_target <= best_score * 1.001:
            continue
        
        # Create residuals function for this target
        residuals_func = _make_residuals(n, r_target)
        
        # Initial guess: current best configuration
        x0 = best_centres.flatten()
        
        # Bounds: centres must stay within reasonable range
        bounds = ([-1.05] * 2 * n, [1.05] * 2 * n)
        
        try:
            result = least_squares(
                residuals_func,
                x0,
                method='trf',
                max_nfev=5000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                bounds=bounds
            )
            
            new_centres = result.x.reshape(n, 2)
            new_score = _compute_score(new_centres)
            
            # Accept if finite and improved
            if np.all(np.isfinite(new_centres)) and new_score > best_score:
                best_centres = new_centres.copy()
                best_score = new_score
        except Exception:
            pass
    
    # Final refinement: try to push slightly beyond best found
    if best_score > 0:
        r_final = best_score * 1.005
        residuals_func = _make_residuals(n, r_final)
        x0 = best_centres.flatten()
        bounds = ([-1.05] * 2 * n, [1.05] * 2 * n)
        
        try:
            result = least_squares(
                residuals_func,
                x0,
                method='trf',
                max_nfev=8000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                bounds=bounds
            )
            
            new_centres = result.x.reshape(n, 2)
            new_score = _compute_score(new_centres)
            
            if np.all(np.isfinite(new_centres)) and new_score > best_score:
                best_centres = new_centres.copy()
                best_score = new_score
        except Exception:
            pass
    
    # Verify final result
    final_score = _compute_score(best_centres)
    if final_score <= 0 or not np.all(np.isfinite(best_centres)):
        # Fallback to hexagonal lattice
        return _hexagonal_initial(n, 0.9)
    
    return best_centres
# EVOLVE-BLOCK-END
