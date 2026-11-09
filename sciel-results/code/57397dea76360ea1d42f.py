"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Constructs the centers of 300 circles inside a unit circle.
    Uses a physics-based relaxation method (active constraint pushing)
    starting from a hexagonal lattice initialization.
    """
    
    # 1. Initialization: Hexagonal Lattice
    # We generate a hexagonal grid and pick the N points closest to the origin
    # that fit inside the unit circle (with a small margin).
    
    def generate_hex_lattice(n_points):
        # Estimate spacing
        # Area of unit circle = pi. Area per hex cell ~ (sqrt(3)/2) * s^2.
        # N * (sqrt(3)/2) * s^2 ~ pi => s ~ sqrt(2*pi/(N*sqrt(3)))
        s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
        
        # Generate points in a square grid large enough to cover the unit circle
        # Hex lattice points:
        # x = (i + 0.5 * (j % 2)) * s
        # y = j * s * sqrt(3) / 2
        max_k = int(2.0 / s) + 2
        points = []
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append([x, y])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        # Keep points well inside the circle initially
        # We want to pick N points. 
        # Sort by distance to origin and take the first N? 
        # This creates a "disk" of points.
        points = points[np.argsort(dists)]
        
        # We might have more than N points if the grid is dense.
        # We need exactly N.
        # The best initial guess is the N points closest to the center.
        return points[:n_points]

    # Initial positions
    positions = generate_hex_lattice(N)
    
    # 2. Optimization: Active Constraint Pushing
    # We want to maximize r = min(min_{i<j} |ci-cj|/2, min_i (1-|ci|))
    # This is equivalent to minimizing the "violation" of the constraints 
    # for a given r, but we don't know r. 
    # Instead, we iteratively push the closest pairs apart.
    
    def compute_forces(pos, margin=0.001, stiffness=1.0):
        """
        Computes repulsive forces for pairs closer than current min_dist + margin
        and attractive forces towards center for points closer to boundary than min_dist + margin.
        """
        n = len(pos)
        forces = np.zeros_like(pos)
        
        # Pairwise distances
        # Vectorized calculation:
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        
        # For N=300, N^2 is 90k. This is fine.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Find current minimum distance (excluding self)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary distances
        radii = np.linalg.norm(pos, axis=1)
        boundary_dists = 1.0 - radii
        min_boundary_dist = np.min(boundary_dists)
        
        # Current effective radius
        r_eff = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # We push pairs that are close to the minimum
        threshold_pair = min_pair_dist + margin
        # Mask for active pairs
        active_mask = (dists < threshold_pair) & (dists > 0)
        
        # Calculate forces for active pairs
        # Force is along the vector connecting them, magnitude proportional to violation
        # Violation = threshold_pair - dist
        # We only want to push if they are "too close" relative to the target expansion
        # Actually, we just want to increase the minimum. 
        # If we push all pairs < threshold, we increase the min.
        
        # Normalize diff vectors
        # dists is 0 on diagonal, inf on diagonal for min calc, but diff is 0.
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-8, dists, 1.0)
        norm_diff = diff / safe_dists[:, :, np.newaxis]
        
        # Magnitudes
        # We want to push apart. 
        # Strength = (threshold - dist)
        strengths = np.where(active_mask, (threshold_pair - dists), 0.0)
        
        # Force on i from j is strengths[i,j] * norm_diff[i,j]
        # But norm_diff[i,j] is (pos[i] - pos[j]) / dist. 
        # So force on i is in direction away from j. Correct.
        force_pairs = strengths[:, :, np.newaxis] * norm_diff
        forces += np.sum(force_pairs, axis=1)
        
        # Boundary forces
        threshold_boundary = min_boundary_dist + margin
        active_boundary_mask = boundary_dists < threshold_boundary
        
        # Force towards center
        # Direction is -pos / |pos|
        safe_radii = np.where(radii > 1e-8, radii, 1.0)
        norm_pos = pos / safe_radii[:, np.newaxis]
        
        strengths_b = np.where(active_boundary_mask, (threshold_boundary - boundary_dists), 0.0)
        
        # Push towards center (negative direction of pos)
        force_boundary = -strengths_b[:, np.newaxis] * norm_pos
        forces += force_boundary
        
        return forces, r_eff

    # Optimization Loop
    # We use a simulated annealing-like schedule for step size
    n_iter = 2000
    step_size = 0.005
    cooling_rate = 0.995
    margin = 0.005
    
    for i in range(n_iter):
        forces, r_eff = compute_forces(positions, margin=margin)
        
        # Update positions
        positions += step_size * forces
        
        # Hard constraint: Keep inside unit circle
        # If any point goes outside, project it back
        radii = np.linalg.norm(positions, axis=1)
        outside_mask = radii > 1.0
        if np.any(outside_mask):
            # Project to boundary
            positions[outside_mask] = positions[outside_mask] / radii[outside_mask, np.newaxis]
        
        # Decay step size and margin
        step_size *= cooling_rate
        margin *= 0.999
        
        # Optional: Stop early if no improvement? 
        # But we need to run enough to beat the lattice.
        
    # Final polish: Small step size refinement
    # The above loop might leave the solution slightly sub-optimal if it stops too early or step size is too small.
    # Let's do a final local search with very small steps to ensure we are at a local max.
    
    # Re-initialize margin for final polish
    margin = 0.0001
    step_size = 0.0001
    for i in range(500):
        forces, r_eff = compute_forces(positions, margin=margin)
        positions += step_size * forces
        radii = np.linalg.norm(positions, axis=1)
        outside_mask = radii > 1.0
        if np.any(outside_mask):
            positions[outside_mask] = positions[outside_mask] / radii[outside_mask, np.newaxis]

    return positions

# EVOLVE-BLOCK-END
