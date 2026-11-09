"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice packing of circles in the unit disk."""
    # Estimate spacing for n_points circles in unit disk
    # Area of disk = pi, area per circle ~ (spacing * sqrt(3)/2) for hex packing
    # n * (s/2)^2 * pi <= pi  =>  s ~ 2/sqrt(n)
    # But hex packing is denser, so s can be a bit larger
    # For hex packing: density = pi/(2*sqrt(3)) * (1/r^2) where r is spacing
    # Actually, let's just use a reasonable estimate and refine
    
    # Start with a spacing estimate
    spacing = 2.0 / np.sqrt(n_points) * 0.95
    
    # Generate hex lattice points
    points = []
    # Hex lattice: rows offset
    row_spacing = spacing * np.sqrt(3) / 2
    # How many rows fit in radius 1?
    n_rows = int(2.0 / row_spacing) + 3
    
    for j in range(-n_rows, n_rows + 1):
        y = j * row_spacing
        x_offset = (spacing / 2) * (j % 2)
        n_cols = int(2.0 / spacing) + 3
        for i in range(-n_cols, n_cols + 1):
            x = (i + x_offset) * spacing
            if np.hypot(x, y) <= 1.0:
                points.append((x, y))
    
    points = np.array(points)
    
    # Sort by distance from origin, take the closest n_points
    if len(points) >= n_points:
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists, kind='stable')[:n_points]
        return points[idx]
    else:
        # If not enough points, add random points
        while len(points) < n_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 1)
            points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
        return points[:n_points]


def _compute_score(centers):
    """Compute the packing score: min(min pairwise dist/2, min boundary clearance)."""
    # Boundary clearance
    radii = np.hypot(centers[:, 0], centers[:, 1])
    boundary_clearance = np.min(1.0 - radii)
    
    # Pairwise distances
    if len(centers) > 1:
        dists = pdist(centers)
        min_pair_dist = np.min(dists)
        pair_clearance = min_pair_dist / 2.0
    else:
        pair_clearance = 1.0
    
    return min(boundary_clearance, pair_clearance)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initialize with hex lattice
    np.random.seed(42)
    centers = _hex_lattice_init(N)
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # SLSQP optimization with constraints
    # Objective: maximize min(pairwise_dist/2, boundary_clearance)
    # Equivalent to: minimize -min(...)
    # But SLSQP works with differentiable objectives, so we use a smooth approximation
    
    # Strategy: use a soft-min function to approximate the minimum
    def soft_min(values, k=10.0):
        """Soft minimum of a set of values."""
        return -np.log(np.sum(np.exp(-k * values))) / k
    
    def objective(x):
        """Negative of the soft-min of all constraints (to maximize)."""
        x = x.reshape(-1, 2)
        
        # Boundary constraint: 1 - |x_i|
        radii = np.hypot(x[:, 0], x[:, 1])
        boundary = 1.0 - radii
        
        # Pairwise distances / 2
        if len(x) > 1:
            dists = pdist(x) / 2.0
        else:
            dists = np.array([1.0])
        
        all_values = np.concatenate([boundary, dists])
        return -soft_min(all_values, k=10.0)
    
    # Constraints for SLSQP
    def constraint_boundary(x):
        """Each circle must be inside the unit disk."""
        x = x.reshape(-1, 2)
        radii = np.hypot(x[:, 0], x[:, 1])
        return 1.0 - radii - 0.001  # Small margin
    
    def constraint_pairwise(x):
        """Pairwise distances must be at least 2*r for some r."""
        # This is tricky - we need to ensure all pairs are separated
        # For SLSQP, we can't have 44850 constraints efficiently
        # Instead, we rely on the objective to push separation
        
        x = x.reshape(-1, 2)
        # Just return a dummy constraint that's always satisfied
        # The real work is done by the objective
        return np.ones(1)
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Run SLSQP optimization
    x0 = centers.flatten()
    
    # Try a few restarts with different initializations
    for restart in range(3):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x0 = centers.flatten()
        else:
            # Perturb the best solution
            x0 = best_centers.flatten() + np.random.normal(0, 0.01, 2 * N)
        
        # Clip to bounds
        x0 = np.clip(x0, -1.0, 1.0)
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.success:
                new_centers = result.x.reshape(-1, 2)
                # Clip to ensure inside disk
                new_radii = np.hypot(new_centers[:, 0], new_centers[:, 1])
                mask = new_radii <= 1.0
                if np.sum(mask) >= N:
                    new_centers = new_centers[mask][:N]
                    new_score = _compute_score(new_centers)
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = new_centers.copy()
        except Exception as e:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # Final check: ensure all points are inside the unit disk
    radii = np.hypot(best_centers[:, 0], best_centers[:, 1])
    mask = radii <= 1.0
    if np.sum(mask) < N:
        # Fallback to original hex lattice
        best_centers = _hex_lattice_init(N)
    
    return best_centers

# EVOLVE-BLOCK-END
