"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def _hex_lattice(n, shrink=0.95):
    """Generate n hexagonal lattice points inside a disk of radius ~shrink."""
    # Estimate spacing for n points in a hexagonal packing
    # Area per hex point ~ sqrt(3)/2 * s^2, total area ~ pi * shrink^2
    # n * sqrt(3)/2 * s^2 ~ pi * shrink^2  =>  s ~ sqrt(2*pi*shrink^2/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi * shrink**2 / (n * np.sqrt(3.0)))
    # Generate hex lattice points
    k = int(1.0 / s) + 3
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= shrink:
                rows.append((x, y))
    points = np.array(rows)
    if len(points) < n:
        # Fallback: densify
        s *= 0.9
        rows = []
        k = int(1.0 / s) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= shrink:
                    rows.append((x, y))
        points = np.array(rows)
    # Sort by radius and take first n
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind="stable")[:n]
    return points[idx]


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares with a smooth inverse-distance objective
    to maximize the minimum radius.
    """
    n = N
    rng = np.random.default_rng(42)
    
    best_radius = 0.0
    best_centres = None
    
    # Multi-start: try several perturbed hexagonal lattices
    n_starts = 5
    for start_idx in range(n_starts):
        # Generate initial hex lattice
        shrink = 0.95 if start_idx == 0 else 0.90 + 0.02 * start_idx
        x0_2d = _hex_lattice(n, shrink=shrink)
        
        # Add small random perturbation to break symmetry and help escape local minima
        if start_idx > 0:
            perturbation = 0.005 * rng.standard_normal((n, 2))
            x0_2d = x0_2d + perturbation
        
        x0 = x0_2d.flatten()
        
        def residuals(x):
            """
            Smooth, differentiable objective: minimize sum of inverse distances.
            For pairs: 1/d_ij (large when close, small when far)
            For boundary: 1/(1 - |c_i|) (large when near boundary)
            
            This is smooth for d_ij > 0 and |c_i| < 1.
            """
            centres = x.reshape(n, 2)
            
            # Pairwise distances
            dists = pdist(centres)
            
            # Boundary distances
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            
            # Clip to avoid division by zero or negative
            eps = 1e-6
            dists_safe = np.maximum(dists, eps)
            boundary_dists_safe = np.maximum(boundary_dists, eps)
            
            # Residuals: inverse distances
            pair_residuals = 1.0 / dists_safe
            boundary_residuals = 1.0 / boundary_dists_safe
            
            return np.concatenate([pair_residuals, boundary_residuals])
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=5000,
                                   xtol=1e-8, ftol=1e-8, gtol=1e-8)
            centres = result.x[:2 * n].reshape(n, 2)
            
            # Compute actual radius
            dists = pdist(centres)
            min_pair_dist = np.min(dists)
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            min_boundary_dist = np.min(boundary_dists)
            r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            # Check validity
            if r > 0 and np.isfinite(r) and np.all(boundary_dists > 0):
                if r > best_radius:
                    best_radius = r
                    best_centres = centres
        except Exception:
            pass
    
    if best_centres is None:
        # Fallback to hexagonal lattice
        return _hex_lattice(n, shrink=0.95)
    
    return best_centres
# EVOLVE-BLOCK-END
