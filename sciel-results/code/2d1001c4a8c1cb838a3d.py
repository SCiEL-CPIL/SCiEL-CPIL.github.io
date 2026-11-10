# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def _generate_topologies(num_candidates=20):
    """
    Generates candidate topologies of 11 unit hexagons on a triangular lattice.
    Uses a greedy growth strategy with different cost functions to ensure diversity.
    Returns a list of arrays, each of shape (11, 2).
    """
    # Lattice basis vectors for unit hexagons (distance between centers is 2)
    # Axial coordinates (q, r) to Cartesian (x, y)
    # x = 2*q + r
    # y = sqrt(3)*r
    sqrt3 = np.sqrt(3)
    
    def axial_to_cart(q, r):
        return np.array([2*q + r, sqrt3*r])

    # Directions to neighbors in axial coordinates
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    topologies = []
    
    # Strategy 1: Minimize max distance from centroid (compactness)
    # Strategy 2: Minimize sum of squared distances (moment of inertia)
    # Strategy 3: Randomized greedy
    
    strategies = ['max_dist', 'variance', 'random']
    
    for strat in strategies:
        for seed in range(5):
            np.random.seed(seed)
            current = [(0, 0)]
            current_set = set(current)
            
            for _ in range(10):
                # Find all valid neighbors
                candidates = []
                for (q, r) in current:
                    for dq, dr in directions:
                        nq, nr = q + dq, r + dr
                        if (nq, nr) not in current_set:
                            candidates.append((nq, nr))
                
                if not candidates:
                    break
                
                # Remove duplicates from candidates
                candidates = list(set(candidates))
                
                best_candidate = None
                best_cost = np.inf
                
                for (nq, nr) in candidates:
                    # Tentatively add
                    new_set = current + [(nq, nr)]
                    # Calculate cost
                    pts = np.array([axial_to_cart(q, r) for q, r in new_set])
                    centroid = pts.mean(axis=0)
                    shifted = pts - centroid
                    
                    if strat == 'max_dist':
                        dists = np.linalg.norm(shifted, axis=1)
                        cost = np.max(dists)
                    elif strat == 'variance':
                        dists = np.linalg.norm(shifted, axis=1)
                        cost = np.var(dists)
                    else:
                        # Random selection, but prefer closer to centroid slightly
                        dists = np.linalg.norm(shifted, axis=1)
                        cost = np.random.rand() * 10 + np.mean(dists)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = (nq, nr)
                
                if best_candidate is not None:
                    current.append(best_candidate)
                    current_set.add(best_candidate)
            
            if len(current) == 11:
                pts = np.array([axial_to_cart(q, r) for q, r in current])
                # Center the points
                centroid = pts.mean(axis=0)
                pts_centered = pts - centroid
                topologies.append(pts_centered)
    
    # Deduplicate topologies (based on sorted points)
    unique_topologies = []
    seen = set()
    for t in topologies:
        # Sort points to create a hashable representation
        sorted_pts = np.sort(t, axis=0)
        key = tuple(map(tuple, sorted_pts))
        if key not in seen:
            seen.add(key)
            unique_topologies.append(t)
            
    return unique_topologies

def _min_enclosing_hex_side(points, angle_deg=0.0):
    """
    Calculates the side length of the minimal regular hexagon with the given orientation
    that encloses the given points.
    points: (N, 2) array.
    angle_deg: Orientation of the hexagon's face normals.
    """
    a = np.deg2rad(angle_deg)
    # Face normals of a regular hexagon
    mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])  # (6, 2)
    
    # Project points onto normals
    # points: (N, 2), normals.T: (2, 6) -> dots: (N, 6)
    dots = points @ normals.T
    # The radius r is the max projection
    r = dots.max()
    # Side length s = 2 * r / sqrt(3)
    return 2.0 * r / np.sqrt(3.0)

def _optimize_rotation(points):
    """
    Finds the optimal rotation angle for the cluster of points to minimize
    the enclosing hexagon side length (assuming the enclosing hexagon is aligned to 0).
    """
    def objective(theta_rad):
        theta_deg = np.rad2deg(theta_rad)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        rotated_points = points @ R.T
        return _min_enclosing_hex_side(rotated_points, angle_deg=0.0)
    
    # Minimize over [0, 60 degrees)
    result = minimize_scalar(objective, bounds=(0, np.deg2rad(60)), method='bounded', options={'xatol': 1e-8})
    best_theta_rad = result.x
    best_side = result.fun
    
    # Also check the exact boundaries just in case (though bounded method handles it)
    # And maybe a few discrete points to be safe against local minima in the rotation
    # The function is periodic and smooth, bounded method should be fine.
    
    return best_theta_rad, best_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a rigid-body cluster approach: generates candidate topologies on a lattice,
    then optimizes the global rotation of the cluster.
    """
    topologies = _generate_topologies(num_candidates=20)
    
    if not topologies:
        # Fallback: simple grid
        pts = np.array([
            [0, 0],
            [2, 0], [1, np.sqrt(3)], [-1, np.sqrt(3)], [-2, 0], [-1, -np.sqrt(3)], [1, -np.sqrt(3)],
            [4, 0], [3, np.sqrt(3)], [-3, np.sqrt(3)], [-4, 0]
        ])
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        topologies = [pts]
        
    best_side = np.inf
    best_points = None
    best_theta = 0.0
    
    for pts in topologies:
        # Center the points
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        
        theta_rad, side = _optimize_rotation(pts_centered)
        
        if side < best_side:
            best_side = side
            best_points = pts_centered
            best_theta = theta_rad
            
    # Rotate the best points by the optimal angle
    c, s = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    final_points = best_points @ R.T
    
    # Construct inner array
    # Each row: (x, y, angle_deg)
    angle_deg = np.rad2deg(best_theta)
    inner = np.column_stack([final_points[:, 0], final_points[:, 1], np.full(11, angle_deg)])
    
    # Construct outer array
    # The outer hexagon is aligned with angle 0 (since we optimized relative to 0)
    outer = np.array([0.0, 0.0, 0.0])
    
    side = float(best_side)
    
    return inner, outer, side

# EVOLVE-BLOCK-END
