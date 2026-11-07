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
    
    # 1. Base Heuristic: Inverse Distance (linear)
    # Using 1/d instead of 1/d^2 reduces the extreme bias towards very close nodes,
    # allowing the angular and demand factors to have a more balanced influence.
    # This can help avoid "traps" where ants get stuck in small clusters.
    eps = 1e-9
    base_heuristic = 1.0 / (dm + eps)
    
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
    # Adjusting lambda to 2.0 to be slightly less restrictive, allowing more flexibility.
    lambda_angle = 2.0
    
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
    
    # 4. Demand Awareness (Pair-wise Compatibility)
    # Instead of a simple per-node penalty, we use a pair-wise compatibility penalty.
    # If two nodes have high demands, their combined demand might exceed capacity,
    # or at least be less "compatible" for routing together if we consider load balancing.
    # However, a simpler and effective heuristic is to penalize edges where the sum of demands is high,
    # as this suggests a "heavy" connection that might be less efficient if not handled carefully.
    # Or, we can use a soft penalty that encourages mixing high and low demand nodes?
    # Actually, the previous finding suggested "Pair-wise demand compatibility penalty ... creates a more nuanced heuristic".
    # Let's try penalizing edges where the sum of demands is close to or exceeds capacity?
    # No, that's too strong. Let's stick to a smooth penalty based on the product or sum.
    # Let's try: penalty = 1 / (1 + 0.05 * (dem[i] + dem[j]))
    # This penalizes edges connecting two high-demand nodes more than edges connecting low-demand nodes.
    
    dem_sum = dem[:, np.newaxis] + dem[np.newaxis, :]
    demand_penalty = 1.0 / (1.0 + 0.05 * dem_sum)
    heuristic *= demand_penalty
    
    # 5. Sanitize
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
