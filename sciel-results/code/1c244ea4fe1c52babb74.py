"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ant colony optimization of CVRP.
    
    This implementation uses a structural change: an Angular Continuity Heuristic.
    It combines a steep inverse-distance gradient (1/d^2) with an exponential 
    penalty for angular deviation from the depot's perspective. This encourages 
    ants to form spatially contiguous, non-crossing routes that respect the 
    angular sectors of the unit square, which is a strong characteristic of 
    optimal CVRP solutions.
    """
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # 1. Base Heuristic: Inverse Squared Distance
    # Steeper gradient encourages tight local clusters, reducing total travel distance.
    # We add epsilon to avoid division by zero (though dm[0,0]=0, we handle it).
    eps = 1e-9
    base_heuristic = 1.0 / (dm + eps)**2
    
    # 2. Angular Continuity Heuristic
    # Calculate the angle of each customer relative to the depot (node 0)
    # coords[0] is the depot.
    depot_coords = coords[0]
    
    # Vectorized angle calculation for all nodes
    dx = coords[:, 0] - depot_coords[0]
    dy = coords[:, 1] - depot_coords[1]
    
    # Angle in radians [-pi, pi]
    angles = np.arctan2(dy, dx)
    
    # We want to penalize transitions between nodes i and j that have large 
    # angular differences. This promotes "sector-based" routing.
    # Create a matrix of angle differences
    # delta_angles[i, j] = |angle_i - angle_j| wrapped to [-pi, pi]
    angle_diff = angles[:, np.newaxis] - angles[np.newaxis, :]
    
    # Wrap to [-pi, pi]
    # (x + pi) % (2*pi) - pi
    angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi
    
    # Exponential decay based on squared angular difference
    # Lambda controls how strongly we penalize angular jumps.
    # Higher lambda = stricter angular continuity.
    lambda_angle = 4.0
    
    angular_factor = np.exp(-lambda_angle * (angle_diff ** 2))
    
    # 3. Combine Heuristics
    # For edges involving the depot (node 0), angular continuity is less 
    # critical for the "route body" structure, but distance is still key.
    # We can apply the angular factor only to customer-customer edges.
    
    heuristic = base_heuristic.copy()
    
    # Apply angular factor only where both i and j are customers (index > 0)
    # Create a mask
    mask = np.ones((n, n), dtype=bool)
    mask[0, :] = False
    mask[:, 0] = False
    
    # Apply the angular penalty
    heuristic[mask] *= angular_factor[mask]
    
    # 4. Demand Awareness (Optional Refinement)
    # High demand nodes are "costly" in terms of capacity.
    # We might slightly penalize visiting high demand nodes if they are far,
    # but the angular/distance terms usually dominate.
    # Let's add a very soft demand penalty to avoid ants ignoring large loads 
    # in favor of many small ones if the geometry is ambiguous.
    # Penalty: 1 / (1 + 0.05 * demand)
    # This is kept very small to not disrupt the geometric structure.
    demand_penalty = 1.0 / (1.0 + 0.05 * dem)
    heuristic *= demand_penalty[np.newaxis, :]
    
    # 5. Sanitize
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
