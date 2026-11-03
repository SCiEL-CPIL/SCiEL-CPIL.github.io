"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares with a smooth objective that pushes r up
    while minimizing constraint violations.
    """
    n = N
    
    # Generate initial hexagonal lattice configuration
    def hex_lattice_init(n_points, max_radius=0.95):
        # Estimate spacing
        area = np.pi * max_radius**2
        hex_area = np.sqrt(3)/2  # area per hex unit for spacing=1
        spacing = np.sqrt(area / (n_points * hex_area))
        
        # Generate hex lattice points
        k = int(np.ceil(max_radius / spacing)) + 1
        points = []
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= max_radius:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n_points:
            # Fallback: random points
            angles = np.random.uniform(0, 2*np.pi, n_points)
            radii = np.random.uniform(0, max_radius, n_points)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Sort by distance from center, take closest n_points
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists, kind='stable')[:n_points]
            points = points[idx]
        
        return points
    
    initial_points = hex_lattice_init(n, max_radius=0.9)
    
    # Initial radius estimate
    dists_init = pdist(initial_points)
    min_pair = np.min(dists_init) / 2.0
    min_boundary = np.min(1.0 - np.linalg.norm(initial_points, axis=1))
    r0 = min(min_pair, min_boundary)
    
    # Variables: [x_1, y_1, ..., x_n, y_n, r]
    x0 = np.concatenate([initial_points.flatten(), [r0]])
    
    # Large penalty weight to encourage large r
    penalty_weight = 50.0
    
    def residuals(x):
        """
        Smooth residuals:
        - For each pair (i,j): 2r - d_ij (want <= 0)
        - For each point i: r - (1 - |c_i|) (want <= 0)
        - Penalty term: -penalty_weight * r (to push r up)
        
        We square all these, so minimizing sum of squares will:
        - Try to make 2r - d_ij <= 0 (increase d_ij or decrease r)
        - Try to make r - (1-|c_i|) <= 0 (decrease r or move inward)
        - Try to maximize r due to the penalty term
        """
        centres = x[:2*n].reshape(n, 2)
        r = x[2*n]
        
        # Pairwise distances
        dists = pdist(centres)
        
        # Pair residuals: 2r - d_ij
        pair_res = 2.0 * r - dists
        
        # Boundary residuals: r - (1 - |c_i|)
        boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
        boundary_res = r - boundary_dists
        
        # Penalty to maximize r
        penalty_res = -penalty_weight * r
        
        return np.concatenate([pair_res, boundary_res, [penalty_res]])
    
    # Run optimization
    try:
        result = least_squares(
            residuals, x0, method='trf',
            max_nfev=5000,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
            bounds=([-1.0]*2*n + [0.0], [1.0]*2*n + [0.5])
        )
        
        centres = result.x[:2*n].reshape(n, 2)
        r_opt = result.x[2*n]
        
    except Exception as e:
        centres = initial_points
        r_opt = r0
    
    # Compute actual achievable radius
    dists = pdist(centres)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    actual_r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Validate
    if not np.isfinite(actual_r) or actual_r <= 0:
        # Fallback: use initial configuration
        centres = initial_points
        dists = pdist(centres)
        min_pair_dist = np.min(dists)
        boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
        min_boundary_dist = np.min(boundary_dists)
        actual_r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    return centres
# EVOLVE-BLOCK-END
