"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Initialize sphere centers using a deterministic Fibonacci lattice
    # to ensure maximal initial separation, then refine positions via
    # scipy.optimize.minimize to maximize the minimum pairwise distance
    # and boundary clearance.
    
    def fibonacci_lattice(n):
        """Generate n points on a unit cube using Fibonacci lattice."""
        points = np.zeros((n, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            # Fibonacci sphere projection mapped to cube
            t = i / n
            # Use a 3D Fibonacci-like distribution
            # We'll use a simple approach: distribute on a 3D grid with Fibonacci spacing
            x = (i * golden_ratio) % 1.0
            y = (i * golden_ratio * golden_ratio) % 1.0
            z = (i * golden_ratio * golden_ratio * golden_ratio) % 1.0
            # Map to [-0.5, 0.5]
            points[i] = [x - 0.5, y - 0.5, z - 0.5]
        return points
    
    def objective(params):
        """Objective to minimize: negative of the minimum clearance."""
        centers = params.reshape(-1, 3)
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Compute boundary clearance
        # Distance from each center to the nearest face of the cube [-0.5, 0.5]^3
        boundary_dists = np.min(np.abs(centers), axis=1)  # distance to nearest face
        # Actually, distance to boundary is 0.5 - max(|x|, |y|, |z|) for each point
        # Wait, the cube is [-0.5, 0.5]^3, so distance to boundary for point c is
        # min(0.5 - |c_x|, 0.5 - |c_y|, 0.5 - |c_z|)
        # But we want the minimum over all points and all faces
        # Distance from center to nearest face: for each axis, dist = 0.5 - abs(coord)
        # The sphere of radius r fits if r <= min over all points of min over all axes of (0.5 - abs(coord))
        # AND r <= min over all pairs of dist/2
        
        # So the common radius is:
        # r = min(min_pair_dist / 2, min_boundary_clearance)
        # where min_boundary_clearance = min_i min_k (0.5 - |c_ik|)
        
        boundary_clearance = np.min(0.5 - np.abs(centers))
        
        r = min(min_pair_dist / 2.0, boundary_clearance)
        
        # We want to maximize r, so minimize -r
        return -r
    
    # Initialize with Fibonacci lattice
    centers_init = fibonacci_lattice(N)
    
    # Flatten for optimization
    params_init = centers_init.flatten()
    
    # Use Nelder-Mead for optimization (no gradient needed, robust)
    result = minimize(objective, params_init, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    # Extract optimized centers
    centers = result.x.reshape(-1, 3)
    
    # Ensure centers are within the cube (clip to valid range)
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
