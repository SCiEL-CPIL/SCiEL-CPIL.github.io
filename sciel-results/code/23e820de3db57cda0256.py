"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Constructs the centers of 300 circles inside a unit circle.
    Uses hexagonal lattice initialization with radial and angular optimization.
    """
    
    def generate_hexagonal_lattice(n_points, scale, rotation, offset_x, offset_y):
        """Generate points on a hexagonal lattice, keep only those within unit circle."""
        # Hexagonal lattice basis vectors
        a1 = scale * np.array([1.0, 0.0])
        a2 = scale * np.array([0.5, np.sqrt(3)/2])
        
        # Generate lattice points in a range
        max_range = 2.0
        i_min, i_max = int(-max_range / scale) - 1, int(max_range / scale) + 1
        j_min, j_max = int(-max_range / scale) - 1, int(max_range / scale) + 1
        
        points = []
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                p = i * a1 + j * a2
                # Apply rotation
                cos_r, sin_r = np.cos(rotation), np.sin(rotation)
                p_rot = np.array([cos_r * p[0] - sin_r * p[1], sin_r * p[0] + cos_r * p[1]])
                # Apply offset
                p_final = p_rot + np.array([offset_x, offset_y])
                # Keep only if within unit circle
                if np.linalg.norm(p_final) <= 1.0:
                    points.append(p_final)
        
        points = np.array(points) if points else np.empty((0, 2))
        
        # If we have more than n_points, randomly select n_points
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
            points = points[indices]
        
        return points

    def calc_radius(pos):
        """Calculate the maximum possible radius for given positions."""
        if len(pos) == 0:
            return 0.0
        
        # Minimum distance between any two circles
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) if len(pos) > 1 else np.inf
        
        # Minimum distance to boundary
        radii = np.linalg.norm(pos, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        # The radius is limited by both
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return max(0.0, r)

    def objective(params):
        """Objective function to minimize (negative of radius)."""
        scale, rotation, offset_x, offset_y = params
        
        # Ensure scale is positive
        scale = max(abs(scale), 0.01)
        
        pos = generate_hexagonal_lattice(N, scale, rotation, offset_x, offset_y)
        
        if len(pos) < N:
            return 1e6  # Penalty for not enough points
        
        r = calc_radius(pos)
        return -r

    # Initial guess: hexagonal lattice with reasonable parameters
    # For 300 circles, estimate scale
    # Area of unit circle = pi, area per circle ~ pi/N
    # Hexagonal packing density ~ 0.9069
    # So effective area per circle ~ pi/(N * 0.9069)
    # Circle area = pi * r^2, so r ~ sqrt(1/(N * 0.9069))
    est_r = np.sqrt(1.0 / (N * 0.9069))
    est_scale = 2 * est_r  # Distance between centers in hex lattice
    
    # Try multiple initial configurations
    best_pos = None
    best_r = 0.0
    
    for trial in range(10):
        # Random initial parameters
        scale = est_scale * np.random.uniform(0.8, 1.2)
        rotation = np.random.uniform(0, 2 * np.pi)
        offset_x = np.random.uniform(-0.1, 0.1)
        offset_y = np.random.uniform(-0.1, 0.1)
        
        pos = generate_hexagonal_lattice(N, scale, rotation, offset_x, offset_y)
        
        if len(pos) < N:
            continue
        
        r = calc_radius(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
    
    # If we didn't get enough points, use a different strategy
    if best_pos is None or len(best_pos) < N:
        # Fallback: random points
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        best_pos = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        best_r = calc_radius(best_pos)
    
    # Now optimize the lattice parameters
    if best_pos is not None and len(best_pos) == N:
        # We need to recover the lattice parameters from the initial best
        # This is tricky, so instead we do a local optimization on the positions directly
        
        def optimize_positions(x):
            pos = x.reshape(N, 2)
            # Ensure all points are within unit circle
            radii = np.linalg.norm(pos, axis=1)
            outside = radii > 1.0
            if np.any(outside):
                pos[outside] = pos[outside] / radii[outside, np.newaxis]
            
            r = calc_radius(pos)
            return -r
        
        x0 = best_pos.flatten()
        
        try:
            res = minimize(optimize_positions, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_r:
                best_pos = res.x.reshape(N, 2)
        except Exception:
            pass
    
    # Final projection to ensure validity
    radii = np.linalg.norm(best_pos, axis=1)
    outside_mask = radii > 1.0
    if np.any(outside_mask):
        best_pos[outside_mask] = best_pos[outside_mask] / radii[outside_mask, np.newaxis]
    
    return best_pos

# EVOLVE-BLOCK-END
