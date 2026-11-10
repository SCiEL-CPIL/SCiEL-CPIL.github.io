# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    time_limit = 600.0
    
    n = 11
    # Initial guess: a compact cluster
    # A hexagonal number 7 cluster has side 2 (radius 2 in terms of hex centers distance? No, distance between centers is sqrt(3) for unit side hexes touching)
    # Let's use a standard hex grid layout.
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) * 2? No.
    # Distance between centers of touching unit hexagons is sqrt(3).
    # Wait, side length s=1. Distance between centers of two touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    # Layout:
    # Row 0: 1 hex
    # Row 1: 2 hexes
    # Row 2: 3 hexes
    # Row 3: 2 hexes
    # Row 4: 1 hex
    # Total: 1+2+3+2+1 = 9. Need 11.
    
    # Let's try a 3-3-3-2 layout or similar.
    # Or just start with a dense random cluster or a known good packing.
    # A common packing for 11 is a hexagon of side 3 (which holds 19) with some removed?
    # Side 3 hexagon holds 1 + 6 + 12 = 19 hexagons.
    # We can remove 8 from the corners/edges.
    
    # Let's define positions on a hex grid.
    # Hex grid coordinates (q, r) to cartesian (x, y) for pointy top?
    # The problem uses angle_deg. Standard hexagon vertices:
    # (cx + s*cos(a + 60i), cy + s*sin(a + 60i)).
    # If a=0, vertices are at 0, 60, 120, 180, 240, 300 degrees.
    # This is a "flat top" hexagon? No, vertex at 0 deg is to the right.
    # Vertices: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866).
    # This is a hexagon with a vertex pointing right.
    
    # Distance between centers of two such hexagons touching:
    # If they touch along a vertical edge?
    # The edge between (0.5, 0.866) and (0.5, -0.866) is vertical at x=0.5.
    # The next hex to the right would have its left edge at x=0.5.
    # Its leftmost vertex is at cx - 1. So cx - 1 = 0.5 => cx = 1.5.
    # So horizontal distance is 1.5.
    
    # If they touch along an angled edge?
    # Edge from (1,0) to (0.5, 0.866).
    # Normal vector?
    # Distance between centers is 2 * apothem = sqrt(3).
    # Direction of normal to that edge?
    # Midpoint of edge: (0.75, 0.433). Vector from center (0,0) to midpoint is (0.75, 0.433).
    # Length is sqrt(0.75^2 + 0.433^2) = sqrt(0.5625 + 0.1875) = sqrt(0.75) = sqrt(3)/2.
    # So the distance between centers is indeed sqrt(3).
    
    # Let's use a coordinate system based on hex grid steps.
    # Step 1: (1.5, 0)
    # Step 2: (0.75, 1.299)  # sqrt(3)/2, 3/2 * sqrt(3)/2?
    # Let's calculate exact coords for a cluster.
    
    # A good starting configuration for 11 hexagons is a hexagon of radius 2 (side 2) which has 7 hexagons, 
    # plus 4 more attached.
    # Radius 2 hexagon (side length 2 in hex grid units) contains 1 + 6 + 12 = 19? No.
    # "Radius" R in hex grid (distance from center to corner in steps) contains 1 + 3R(R+1).
    # R=1: 1 + 3(1)(2) = 7.
    # R=2: 1 + 3(2)(3) = 19.
    
    # We want 11.
    # Start with R=1 (7 hexes).
    # Add 4 hexes to the outside.
    
    # Center: (0,0)
    # 6 neighbors at distance sqrt(3) from center.
    # Angles: 0, 60, 120, 180, 240, 300.
    # Positions:
    # 0 deg: (sqrt(3), 0)
    # 60 deg: (sqrt(3)/2, 3/2)  # cos(60)=0.5, sin(60)=sqrt(3)/2. dist=sqrt(3). x=sqrt(3)*0.5, y=sqrt(3)*sqrt(3)/2 = 1.5.
    # Wait. sqrt(3) * cos(60) = sqrt(3)/2. sqrt(3) * sin(60) = sqrt(3)*sqrt(3)/2 = 1.5.
    # So (0.866, 1.5).
    
    # Let's generate the 7 center hexes.
    centers = np.array([
        [0.0, 0.0],
        [np.sqrt(3), 0.0],
        [np.sqrt(3)/2, 1.5],
        [-np.sqrt(3)/2, 1.5],
        [-np.sqrt(3), 0.0],
        [-np.sqrt(3)/2, -1.5],
        [np.sqrt(3)/2, -1.5]
    ])
    
    # Add 4 more.
    # Attach to the "corners" or "edges" of the R=1 cluster.
    # The R=1 cluster has 6 "outer" edges on each of the 6 surrounding hexes?
    # Let's just pick 4 positions that extend the shape symmetrically or semi-symmetrically.
    # Positions at distance 2*sqrt(3) from center?
    # Or attached to the outer faces of the 6 neighbors.
    # The neighbor at (sqrt(3), 0) has an outer face facing away from center.
    # Its center is (sqrt(3), 0). The face opposite to the center is at x = sqrt(3) + 1.5?
    # No, the distance between centers is sqrt(3).
    # If we place a hex touching the neighbor at (sqrt(3), 0) on its far side,
    # the new center would be at (sqrt(3) + 1.5, 0)?
    # Let's check.
    # Hex A at (0,0). Hex B at (sqrt(3), 0).
    # Hex C touches B, away from A.
    # Vector A->B is (sqrt(3), 0).
    # Vector B->C should be in the same direction?
    # If so, C is at (2*sqrt(3), 0).
    # Distance B-C is sqrt(3).
    # Do A and C touch? Distance 2*sqrt(3) > sqrt(3). No.
    # Do A and C overlap? No.
    
    # Let's add 4 hexes at:
    # 1. (2*sqrt(3), 0)
    # 2. (sqrt(3), 3.0)  # Above the top-right neighbor?
    #    Neighbor at (sqrt(3)/2, 1.5).
    #    Top face?
    #    Let's just use a grid of candidates and pick 4 that are valid.
    
    # Candidate positions for the 4 extra hexes:
    # Far right: (2*sqrt(3), 0)
    # Far left: (-2*sqrt(3), 0)
    # Top right: (1.5*sqrt(3)/2, 1.5 + 1.5) = (0.75*sqrt(3), 3.0)
    #    Check distance to (sqrt(3)/2, 1.5):
    #    dx = 0.25*sqrt(3), dy = 1.5.
    #    dist^2 = 0.0625*3 + 2.25 = 0.1875 + 2.25 = 2.4375.
    #    sqrt(2.4375) approx 1.56.
    #    sqrt(3) approx 1.732.
    #    So they are close but not touching?
    #    Wait, if they are not touching, we can move them closer to reduce the bounding box.
    #    But for initialization, we just need them to be disjoint.
    
    # Let's use a simpler initialization:
    # A 3x3 grid of hexes would be 9.
    # Add 2 more.
    
    # Actually, let's just use a random initialization in a small box and let the optimizer fix it.
    # This is more robust than trying to guess a perfect topology.
    
    np.random.seed(42)
    # Random positions in a box of size 5x5
    centers = np.random.rand(n, 2) * 5 - 2.5
    
    # Optimization
    # We want to minimize the side length of the enclosing regular hexagon.
    # The enclosing hexagon is centered at (0,0) for simplicity (we can shift the cluster later or optimize center too).
    # Let's optimize the center of the outer hexagon as well.
    # Variables:
    # x, y for each of the 11 hexes (22 vars)
    # cx, cy for the outer hexagon (2 vars)
    # Total 24 vars.
    
    # Objective: Minimize side length S such that all inner hexes are inside the outer hexagon of side S.
    # Constraint: Inner hexes do not overlap.
    
    # How to calculate the minimum S for a given configuration?
    # For a regular hexagon centered at (cx, cy) with side S, the distance from center to any edge (apothem) is S * sqrt(3) / 2.
    # The condition for a point (x, y) to be inside the hexagon is:
    # The maximum of the projections of (x-cx, y-cy) onto the 6 normal vectors of the hexagon edges must be <= S * sqrt(3) / 2.
    # The normals are at angles 0, 60, 120, 180, 240, 300 degrees (if vertex at 0).
    # Wait, the hexagon in the problem has vertices at a + 60i.
    # If a=0, vertices at 0, 60...
    # The edges are perpendicular to the angles 30, 90, 150, 210, 270, 330.
    # So the normals are at 30, 90, 150, 210, 270, 330.
    
    # For an inner hexagon, we need ALL its vertices to be inside the outer hexagon.
    # Since the inner hexagon is convex and the outer is convex, it suffices to check the vertices of the inner hexagon.
    
    # So, for a given set of inner centers and the outer center,
    # S_min = max over all inner vertices v of ( max over 6 normals n of (dot(v - outer_center, n)) ) / (sqrt(3)/2)
    # Wait, the projection must be <= S * sqrt(3) / 2.
    # So S >= 2/sqrt(3) * max_projection.
    
    # We also need to ensure inner hexes don't overlap.
    # Two unit hexagons overlap if the distance between their centers is less than sqrt(3) (if aligned).
    # But they can be rotated.
    # The problem allows rotation of inner hexes?
    # "inner of shape (11, 3), rows (x, y, angle_deg)"
    # Yes, angle is a variable.
    # To minimize complexity, let's fix the angle of all inner hexes to 0 initially, and maybe optimize it later if needed.
    # Or just fix angle to 0. If the optimal packing requires rotation, we might miss it, but 0 is a good start.
    # Actually, for regular hexagons, if they are aligned (same angle), the overlap condition is simple.
    # If they are not aligned, it's complex.
    # Let's assume all inner hexes have angle 0.
    
    # Overlap check for two hexes with angle 0:
    # They overlap if the distance between centers is < sqrt(3).
    # Is this true?
    # Yes, the minimum distance between centers of two non-overlapping unit hexagons with the same orientation is sqrt(3).
    # If they are rotated, the minimum distance can be smaller?
    # No, the "width" of a hexagon is 2 (vertex to vertex) or sqrt(3) (flat to flat).
    # The Minkowski sum of two hexagons determines the exclusion zone.
    # For two identical convex polygons, the minimum distance between centers for non-overlap is the minimum width of the Minkowski sum?
    # Actually, if we fix the orientation, the exclusion zone is a hexagon.
    # If we allow rotation, the exclusion zone is the union of all rotated Minkowski sums.
    # For regular hexagons, the Minkowski sum of two unit hexagons (same orientation) is a regular hexagon with side 2.
    # The distance from center to vertex is 2.
    # So if the distance between centers is > 2, they definitely don't overlap.
    # But they can be closer if they are not aligned?
    # Actually, for regular hexagons, the maximum "reach" is the vertex-to-vertex distance, which is 2.
    # So if distance > 2, no overlap.
    # If distance < sqrt(3), they definitely overlap (since sqrt(3) is the flat-to-flat distance, which is the minimum width).
    # Wait.
    # Flat-to-flat distance is sqrt(3).
    # If two hexagons are aligned and separated by sqrt(3), they touch.
    # If distance < sqrt(3), they overlap.
    # If we rotate one, can we fit them closer?
    # No, the width of a regular hexagon is constant in all directions? No.
    # The width varies between sqrt(3) (flat-to-flat) and 2 (vertex-to-vertex).
    # So the minimum distance to avoid overlap is at least sqrt(3).
    # And it can be up to 2 if they are oriented unfavorably.
    # So, a safe constraint is distance >= 2.
    # A tighter constraint is distance >= sqrt(3) IF they are aligned.
    # Let's use distance >= 2 as a simple constraint to ensure validity, even if it's not optimal.
    # This will lead to a larger outer hexagon, but it's a valid packing.
    # We can try to tighten this later.
    
    # Let's define the optimization function.
    
    def get_outer_side_length(centers, outer_center):
        # centers: (11, 2)
        # outer_center: (2,)
        # Calculate max projection of all inner vertices onto the 6 normals of the outer hexagon.
        
        # Normals of outer hexagon (angle 0 assumed for outer hexagon, or we can fix outer angle to 0)
        # Vertices of outer hexagon at 0, 60, ...
        # Normals at 30, 90, 150, 210, 270, 330.
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Inner hexagon vertices (angle 0)
        inner_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles])
        
        max_proj = 0.0
        for i in range(n):
            cx, cy = centers[i]
            verts = inner_verts_rel + np.array([cx, cy])
            # Shift by outer center
            verts_shifted = verts - outer_center
            # Project onto normals
            projs = verts_shifted @ normals.T
            max_proj = max(max_proj, np.max(np.max(projs)))
            
        # S * sqrt(3)/2 >= max_proj
        # S >= max_proj * 2 / sqrt(3)
        return max_proj * 2 / np.sqrt(3)
    
    def check_overlap(centers):
        # Check if any pair of centers is too close.
        # Use distance >= 2.0 as a safe constraint.
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2.0:
                    return False
        return True
    
    def objective(params):
        # params: [x1, y1, ..., x11, y11, cx, cy]
        centers = params[:2*n].reshape(n, 2)
        outer_center = params[2*n:2*n+2]
        
        side = get_outer_side_length(centers, outer_center)
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2.0:
                    penalty += (2.0 - d)**2
        
        return side + 100 * penalty
    
    # Initial parameters
    params0 = np.concatenate([centers.flatten(), [0, 0]])
    
    # Use scipy.optimize
    from scipy.optimize import minimize
    
    best_params = params0
    best_score = objective(params0)
    
    # Time check
    if time.time() - start_time > time_limit - 10:
        return _format_result(best_params)
        
    # Run optimization
    # L-BFGS-B is good for box constraints, but we have no explicit box.
    # Nelder-Mead might be more robust for non-smooth objectives (due to max function).
    # Let's try Nelder-Mead first.
    
    try:
        res = minimize(objective, params0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
        
    if time.time() - start_time > time_limit - 10:
        return _format_result(best_params)
        
    # Try L-BFGS-B as well, starting from the best found
    try:
        res2 = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 1000})
        if res2.fun < best_score:
            best_params = res2.x
            best_score = res2.fun
    except Exception:
        pass

    return _format_result(best_params)

def _format_result(params):
    n = 11
    centers = params[:2*n].reshape(n, 2)
    outer_center = params[2*n:2*n+2]
    
    # Calculate side length
    # We need to re-calculate the side length based on the final configuration
    # to ensure it's tight enough.
    
    # Normals of outer hexagon
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Inner hexagon vertices (angle 0)
    inner_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    inner_verts_rel = np.array([[np.cos(a), np.sin(a)] for a in inner_angles])
    
    max_proj = 0.0
    for i in range(n):
        cx, cy = centers[i]
        verts = inner_verts_rel + np.array([cx, cy])
        verts_shifted = verts - outer_center
        projs = verts_shifted @ normals.T
        max_proj = max(max_proj, np.max(np.max(projs)))
        
    side = max_proj * 2 / np.sqrt(3)
    
    # Add a small epsilon to side to ensure strict containment if needed?
    # The validator checks cross product >= -1e-6.
    # If we are exactly on the boundary, it might fail due to floating point.
    # Let's add a tiny margin.
    side += 1e-5
    
    # Construct inner hex data
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, 0] = centers[:, 0]
    inner_hex_data[:, 1] = centers[:, 1]
    inner_hex_data[:, 2] = 0.0  # angle
    
    # Construct outer hex data
    outer_hex_data = np.array([outer_center[0], outer_center[1], 0.0])
    
    return inner_hex_data, outer_hex_data, float(side)

# EVOLVE-BLOCK-END
