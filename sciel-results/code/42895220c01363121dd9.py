import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate candidate sets of hexagons on a hexagonal lattice
    # Hexagonal lattice basis vectors for unit hexagons (side 1):
    # Distance between centers in a tight packing is 2 * (sqrt(3)/2) = sqrt(3)
    # Let's use axial coordinates (q, r).
    # Position = q * vec1 + r * vec2
    # vec1 = (sqrt(3), 0)
    # vec2 = (sqrt(3)/2, 1.5)  # Wait, standard hex grid spacing for side 1:
    # Center distance = 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Let's define basis vectors:
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    # This forms a rhombic lattice with 60 degree angles.
    
    sqrt3 = math.sqrt(3)
    u = np.array([sqrt3, 0.0])
    v = np.array([sqrt3/2.0, 1.5])
    
    best_score = float('inf')
    best_result = None
    
    # We want to pack N=11 hexagons.
    # A compact shape is a hexagon of side 2 (which has 1+6+12... no, side 2 hex has 1+6+12? 
    # Side 1: 1
    # Side 2: 1 + 6 = 7
    # Side 3: 1 + 6 + 12 = 19
    # So 11 hexagons fits in a side 2 hexagon (7) plus 4 more.
    # Or a "flower" shape.
    
    # Let's generate all connected subsets of size N from a grid of size ~5x5.
    # To keep it fast, we will try specific "prototypes" or use a simple greedy/beam search
    # if we had time, but for this deterministic step, we will try all subsets of a small grid
    # that are "convex-ish" or just all subsets of a 5x5 grid that contain the center.
    
    # Grid range
    R = 3
    grid_points = []
    for q in range(-R, R+1):
        for r in range(-R, R+1):
            grid_points.append((q, r))
            
    # We need to choose N points from grid_points.
    # Total points in R=3 is 49. 49 choose 11 is huge.
    # We must restrict to "compact" shapes.
    # Strategy: Start with the N most central points. Then try swapping peripheral points.
    
    # Calculate distances from origin for all grid points
    def get_coords(q, r):
        return q * u + r * v

    coords = [get_coords(q, r) for q, r in grid_points]
    dists = [np.linalg.norm(c) for c in coords]
    
    # Sort indices by distance to center
    sorted_indices = np.argsort(dists)
    
    # Initial set: N closest points
    initial_set = set(sorted_indices[:N])
    
    # Evaluate a set of indices
    def evaluate_set(indices):
        if len(indices) != N:
            return float('inf'), None
            
        pts = [coords[i] for i in indices]
        
        # Calculate the required outer hexagon side length.
        # The outer hexagon must contain all these points.
        # The optimal outer hexagon orientation might vary, but let's assume the outer hexagon
        # is aligned with the lattice (angle 0) first, as that's likely optimal for lattice packs.
        # If alignment 0 is not optimal, we might miss the best, but it's a strong baseline.
        
        # The "width" of a set of points in a hexagonal container of side L centered at origin
        # is determined by the maximum projection onto the 6 normal directions.
        # Normals are at angles 0, 60, 120, 180, 240, 300.
        # Actually, for a regular hexagon of side L, the distance from center to side (apothem) is L * sqrt(3) / 2.
        # The condition for a point P to be inside is:
        # |P dot n_k| <= L * sqrt(3) / 2 for k=0..5, where n_k are unit normals.
        
        # Let's compute the required L for a given orientation of the outer hex.
        # We will try outer hex orientation 0 degrees (aligned with lattice).
        
        # Normals for hexagon with flat top (vertices at 0, 60...)?
        # If vertices are at angles 0, 60..., normals are at 30, 90, ...
        # If sides are horizontal/vertical-ish?
        # Standard hexagon: vertices at (L, 0), (L/2, L*sqrt(3)/2)...
        # Let's use the property:
        # For a hexagon centered at origin with side L and rotation theta,
        # the support function in direction phi is L * (cos(phi - theta) + 0.5) ? No.
        
        # Simpler:
        # A regular hexagon of side L is the intersection of 6 strips.
        # The width in any direction is between L*sqrt(3) (across flats) and 2L (across corners).
        # Specifically, if we align the outer hex such that its sides are parallel to the lattice vectors?
        # The lattice vectors are at 0 and 60 degrees.
        # If the outer hex has sides parallel to the lattice vectors, its normals are at 30, 90, 150...
        
        # Let's calculate the maximum projection of the points onto the 6 normal directions
        # for an outer hexagon rotated by 0 degrees.
        # What is "0 degrees"? Let's say one vertex is at (L, 0).
        # Then normals are at 30, 90, 150, 210, 270, 330 degrees.
        
        angles = [30, 90, 150, 210, 270, 330]
        max_proj = 0.0
        for ang_deg in angles:
            ang = math.radians(ang_deg)
            n = np.array([math.cos(ang), math.sin(ang)])
            projs = [np.dot(p, n) for p in pts]
            # The hexagon extends from -L*sqrt(3)/2 to L*sqrt(3)/2 along the normal?
            # No.
            # For a hexagon with vertex at (L,0), the apothem is L*sqrt(3)/2.
            # The normal direction is perpendicular to the side.
            # The side opposite to the vertex at 0 is at 180. Its normal is at 180.
            # The distance from center to that side is L*sqrt(3)/2.
            # So if a point has projection P on normal n, we need P <= L*sqrt(3)/2 and -P <= L*sqrt(3)/2.
            # So |P| <= L*sqrt(3)/2.
            # Therefore L >= |P| * 2 / sqrt(3).
            
            max_abs_proj = max(abs(p) for p in projs)
            required_L = max_abs_proj * 2.0 / sqrt3
            if required_L > max_proj:
                max_proj = required_L
        
        # We must also ensure the hexagon is large enough to contain the points in the "corner" directions?
        # The strip intersection definition covers the entire convex shape.
        # So L = max_proj is the minimal side length for this orientation.
        
        # Let's also try rotating the outer hexagon by 30 degrees (vertices at 30, 90...)
        # Normals would be at 0, 60, 120...
        angles_rot = [0, 60, 120, 180, 240, 300]
        max_proj_rot = 0.0
        for ang_deg in angles_rot:
            ang = math.radians(ang_deg)
            n = np.array([math.cos(ang), math.sin(ang)])
            projs = [np.dot(p, n) for p in pts]
            max_abs_proj = max(abs(p) for p in projs)
            required_L = max_abs_proj * 2.0 / sqrt3
            if required_L > max_proj_rot:
                max_proj_rot = required_L
        
        # The optimal L is the minimum of these? No, we can choose the orientation.
        # So L_min = min(max_proj, max_proj_rot) ? 
        # Actually, we can rotate continuously. But 0 and 30 are the symmetry points.
        # The function L(theta) is periodic with 60 deg symmetry.
        # The minimum is likely at one of the symmetry axes.
        
        L_candidate = min(max_proj, max_proj_rot)
        
        return L_candidate, indices

    # Initial evaluation
    score, _ = evaluate_set(initial_set)
    if score < best_score:
        best_score = score
        best_result = initial_set.copy()
        
    # Local search: Swap one point in the set with one point outside the set
    # If time permits.
    # To be safe, we will perform a limited number of swaps.
    
    num_swaps = 200
    current_set = list(initial_set)
    current_score = score
    
    # Precompute all points not in set
    all_indices = set(range(len(grid_points)))
    
    for _ in range(num_swaps):
        if time.time() > deadline:
            break
            
        # Pick a random index in current_set and a random index not in current_set
        in_idx = np.random.choice(current_set)
        out_candidates = [i for i in all_indices if i not in current_set]
        if not out_candidates:
            break
        out_idx = np.random.choice(out_candidates)
        
        new_set = current_set.copy()
        new_set.remove(in_idx)
        new_set.append(out_idx)
        
        new_score, _ = evaluate_set(new_set)
        
        if new_score < current_score - 1e-9:
            current_set = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_result = current_set.copy()
        # We don't accept worse solutions to keep it simple (Hill Climbing)
        
    if best_result is None:
        # Fallback
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 100.0, 0.0

    # Construct the final output
    inner = []
    for idx in best_result:
        q, r = grid_points[idx]
        x, y = coords[idx]
        # Rotation 0.0
        inner.append([x, y, 0.0])
        
    # Outer hexagon center is (0,0) because we centered the lattice at 0.
    # Outer hexagon side length is best_score.
    # Outer hexagon rotation: We need to return the rotation.
    # We determined L for two orientations. Which one is best?
    # We should return the orientation that achieves best_score.
    
    # Let's re-verify which orientation gave best_score.
    pts = [coords[i] for i in best_result]
    
    # Check orientation 0 (normals at 30...)
    max_proj_0 = 0.0
    for ang_deg in [30, 90, 150, 210, 270, 330]:
        ang = math.radians(ang_deg)
        n = np.array([math.cos(ang), math.sin(ang)])
        projs = [np.dot(p, n) for p in pts]
        max_abs_proj = max(abs(p) for p in projs)
        required_L = max_abs_proj * 2.0 / sqrt3
        if required_L > max_proj_0:
            max_proj_0 = required_L
            
    # Check orientation 30 (normals at 0...)
    max_proj_30 = 0.0
    for ang_deg in [0, 60, 120, 180, 240, 300]:
        ang = math.radians(ang_deg)
        n = np.array([math.cos(ang), math.sin(ang)])
        projs = [np.dot(p, n) for p in pts]
        max_abs_proj = max(abs(p) for p in projs)
        required_L = max_abs_proj * 2.0 / sqrt3
        if required_L > max_proj_30:
            max_proj_30 = required_L
            
    if max_proj_0 <= max_proj_30:
        outer_rotation = 0.0
        final_L = max_proj_0
    else:
        outer_rotation = 30.0
        final_L = max_proj_30
        
    # It's possible a slight rotation is better, but 0 or 30 is a very strong candidate for lattice packs.
    # Given the constraints, we return this.
    
    return inner, (0.0, 0.0), final_L, outer_rotation
# EVOLVE-BLOCK-END
