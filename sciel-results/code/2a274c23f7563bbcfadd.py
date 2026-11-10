# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a tight packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a compact initial arrangement (central flower + 4 extensions) and optimizes positions
    to minimize the outer hexagon side length.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    
    # 1. Define initial compact configuration
    # Unit hexagons (side 1) have circumradius 1.
    # Distance between centers of touching hexagons is 2.
    # 
    # Configuration:
    # 1 central at (0,0)
    # 6 surrounding at distance 2, angles 0, 60, ..., 300
    # 4 extensions in the "gaps" of the flower, pointing outwards.
    # The flower of 7 hexagons has a "radius" (center to farthest vertex) of 2 (center) + 1 (hex) = 3.
    # Actually, the distance from center to vertex of surrounding hex is 2+1=3.
    # The gaps in the 6-neighbor ring are at angles 30, 90, 150, 210, 270, 330.
    # Distance from center to the "corner" of the surrounding hexes is 2 * sin(60) + 1? No.
    # Let's calculate the position for the 4 extra hexagons.
    # They should fit in the V-shaped gaps between adjacent surrounding hexes.
    # The gap is between hex at 0 deg and hex at 60 deg.
    # The bisector is at 30 deg.
    # The surrounding hexes touch at distance 2.
    # The gap vertex is at distance 2 from the center of the surrounding hexes?
    # Let's use a known tight packing for 11 hexagons.
    # A common optimal or near-optimal packing for 11 hexagons in a hexagon:
    # It's essentially a hexagon of side 3 (19 hexes) with 8 removed?
    # Or a hexagon of side 2 (7 hexes) + 4 more?
    
    # Let's try the "7 + 4" configuration.
    # Centers of the 6 surrounding:
    # (2, 0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3))
    
    # The 4 extra hexagons can be placed at the "corners" of this 7-hex cluster.
    # The 7-hex cluster fits in a hexagon of side 3?
    # Vertices of the 7-hex cluster:
    # Farthest points are vertices of the outer hexes.
    # Hex at (2,0) has vertex at (3,0).
    # Hex at (1, sqrt(3)) has vertex at (1+1, sqrt(3)+sqrt(3)/2) = (2, 1.5sqrt(3))? No.
    # Vertex of hex at (cx,cy) with angle 0: (cx+1, cy).
    # Vertex of hex at (1, sqrt(3)) with angle 0: (2, sqrt(3)).
    # The outer boundary is determined by the convex hull of all vertices.
    
    # Let's define the 11 centers.
    sqrt3 = np.sqrt(3)
    
    # 1 central
    centers = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
    ])
    
    # 4 extra. Where to put them?
    # If we put them at the "corners" of the hexagonal shape formed by the 7.
    # The 7 hexes form a hexagon of side 2 (in terms of hex units? No, side 1 hexes).
    # The distance from center to the "midpoint" of the outer edge of the 7-hex cluster is 2.
    # The distance to the "corner" is 2 * cos(30) + 1?
    # Let's just place them in the gaps.
    # Gap 1: between (2,0) and (1,sqrt3). Angle 30 deg.
    # Distance from origin?
    # The touching point of (2,0) and (1,sqrt3) is at (1.5, sqrt3/2).
    # The new hex should be centered further out.
    # If we place it such that it touches both, its center will be at distance 2 from both.
    # Intersection of circles radius 2 around (2,0) and (1,sqrt3).
    # One intersection is (0,0) (the center). The other is the new center.
    # Midpoint of (2,0) and (1,sqrt3) is (1.5, sqrt3/2).
    # Vector from mid to (2,0) is (0.5, -sqrt3/2). Length 1.
    # Perpendicular vector is (sqrt3/2, 0.5). Length 1.
    # Distance from mid to center = sqrt(2^2 - 1^2) = sqrt(3).
    # So new center = Mid + sqrt(3) * (sqrt3/2, 0.5) = (1.5 + 3/2, sqrt3/2 + sqrt3/2) = (3, sqrt3).
    # Let's check distance from (3, sqrt3) to (2,0): sqrt(1 + 3) = 2. Correct.
    # Let's check distance from (3, sqrt3) to (1, sqrt3): sqrt(4 + 0) = 2. Correct.
    
    # So the 4 extra hexagons are at:
    # (3, sqrt3)  -> Gap between 0 and 60
    # (-3, sqrt3) -> Gap between 120 and 180 (Wait, 120 is (-1,sqrt3), 180 is (-2,0)).
    # Let's calculate for 120 and 180.
    # Mid: (-1.5, sqrt3/2). Vec to (-1,sqrt3) is (0.5, sqrt3/2). Perp: (-sqrt3/2, 0.5).
    # Center: (-1.5, sqrt3/2) + sqrt(3)*(-sqrt3/2, 0.5) = (-1.5-1.5, sqrt3/2+sqrt3/2) = (-3, sqrt3).
    # Distance to (-1,sqrt3): sqrt(4+0)=2. Correct.
    
    # So the 4 are at:
    # (3, sqrt3), (-3, sqrt3), (-3, -sqrt3), (3, -sqrt3).
    
    extra_centers = np.array([
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    
    centers = np.vstack([centers, extra_centers])
    
    # All angles are 0
    angles = np.zeros(11)
    
    # Initial positions
    x0 = np.concatenate([centers.flatten(), angles])
    
    # Outer hexagon centered at (0,0), angle 0.
    # We will optimize the outer hexagon's side length S and position (though symmetry suggests 0,0).
    # Let's fix outer center at (0,0) and angle 0 for now, and just optimize S and inner positions.
    # Actually, optimizing inner positions is key to tightening the fit.
    
    def objective(params):
        # params: [x1, y1, a1, ..., x11, y11, a11, S_outer]
        # We want to minimize S_outer.
        # Constraints:
        # 1. Inner hexes don't overlap.
        # 2. Inner hexes are inside outer hex.
        
        # For speed, we will use a penalty method or just minimize S_outer
        # subject to constraints handled by a feasibility check.
        # However, scipy minimize doesn't handle constraints easily for this many.
        # Let's use a penalty function.
        
        inner_xy = params[:22].reshape(11, 2)
        inner_a = params[22:33]
        S = params[33]
        
        # Penalty for S being too small (i.e., if hexes are outside)
        # Penalty for overlap.
        
        penalty = 0.0
        
        # Check containment
        # Outer hex vertices
        outer_verts = np.array([
            [S * np.cos(np.deg2rad(60*i)), S * np.sin(np.deg2rad(60*i))] for i in range(6)
        ])
        
        # For each inner hex, check if any vertex is outside outer hex.
        # This is expensive. 
        # Approximation: Check if distance from center of inner hex to boundary > 0.
        # Or simpler: Just check if the "bounding circle" of inner hex is inside? No, too loose.
        
        # Let's use a simpler heuristic for the optimizer:
        # Minimize the maximum distance from the origin to any vertex of any inner hex.
        # If we minimize the "radius" of the cluster, we can then calculate the minimal S.
        # This decouples the optimization:
        # 1. Optimize inner positions to minimize cluster radius R_max.
        # 2. Calculate minimal S for that R_max.
        
        # Let's redefine objective: Minimize R_max.
        
        max_dist = 0.0
        for i in range(11):
            cx, cy = inner_xy[i]
            a_deg = inner_a[i]
            a_rad = np.deg2rad(a_deg)
            # Vertices of unit hex
            for j in range(6):
                ang = a_rad + np.deg2rad(60*j)
                vx = cx + np.cos(ang)
                vy = cy + np.sin(ang)
                dist = np.sqrt(vx**2 + vy**2)
                if dist > max_dist:
                    max_dist = dist
        
        # Penalty for overlap
        # Check all pairs
        for i in range(11):
            for j in range(i+1, 11):
                # Calculate minimum distance between two hexes
                # If distance < 2, they might overlap.
                # Exact check is complex. Use center distance as proxy?
                # Center distance < 2 implies overlap for hexes with same orientation.
                # With different orientations, it's tricky.
                # Let's assume we keep orientations fixed at 0 for now to simplify.
                # If orientations are fixed, min distance between centers must be >= 2.
                dx = inner_xy[i,0] - inner_xy[j,0]
                dy = inner_xy[i,1] - inner_xy[j,1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        
        return max_dist + 100.0 * penalty
    
    # We will fix angles to 0 to reduce dimensionality and complexity.
    # This is a strong assumption, but for a first improvement, it's viable.
    # If the optimal solution requires rotation, this might be suboptimal, but likely better than side=8.
    
    x0_fixed = np.concatenate([centers.flatten(), np.zeros(1)]) # Last element is dummy, we don't optimize S yet
    
    # Actually, let's just optimize positions (x,y) with angles fixed to 0.
    # Objective: Minimize max_dist + penalty for overlap.
    
    def objective_fixed_angle(params):
        inner_xy = params.reshape(11, 2)
        
        max_dist = 0.0
        for i in range(11):
            cx, cy = inner_xy[i]
            # Vertices of unit hex at angle 0
            # (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
            verts = np.array([
                [1, 0],
                [0.5, sqrt3/2],
                [-0.5, sqrt3/2],
                [-1, 0],
                [-0.5, -sqrt3/2],
                [0.5, -sqrt3/2]
            ]) + inner_xy[i]
            
            dists = np.linalg.norm(verts, axis=1)
            max_dist = max(max_dist, np.max(dists))
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                dx = inner_xy[i,0] - inner_xy[j,0]
                dy = inner_xy[i,1] - inner_xy[j,1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        
        return max_dist + 100.0 * penalty

    # Initial guess
    x0 = centers.flatten()
    
    # Optimize
    try:
        res = minimize(objective_fixed_angle, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 20000})
        
        if res.success:
            opt_centers = res.x.reshape(11, 2)
        else:
            # Fallback to initial if optimization fails
            opt_centers = centers
            
    except Exception:
        opt_centers = centers
        
    # Calculate the minimal outer hexagon side length S for the optimized configuration.
    # The outer hexagon is centered at (0,0) with angle 0.
    # The distance from center to an edge of a regular hexagon with side S is apothem = S * sqrt(3)/2.
    # The distance from center to a vertex is S.
    # For a point (x,y) to be inside the outer hexagon (angle 0):
    # It must satisfy:
    # |x| <= S
    # |y| <= S * sqrt(3)/2
    # |x + y/sqrt(3)| <= S ? No.
    # The 6 inequalities for a hexagon with vertices at (S,0), (S/2, S*sqrt3/2), etc:
    # 1. x <= S
    # 2. -x/2 + y*sqrt(3)/2 <= S
    # 3. -x/2 - y*sqrt(3)/2 <= S
    # 4. -x <= S
    # 5. x/2 - y*sqrt(3)/2 <= S
    # 6. x/2 + y*sqrt(3)/2 <= S
    
    # We need to find the smallest S such that all vertices of all inner hexagons satisfy these.
    # For a given point (x,y), the required S is the maximum of the 6 values:
    # s1 = x
    # s2 = -x/2 + y*sqrt(3)/2
    # s3 = -x/2 - y*sqrt(3)/2
    # s4 = -x
    # s5 = x/2 - y*sqrt(3)/2
    # s6 = x/2 + y*sqrt(3)/2
    
    min_S = 0.0
    for i in range(11):
        cx, cy = opt_centers[i]
        verts = np.array([
            [1, 0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1, 0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ]) + np.array([cx, cy])
        
        for vx, vy in verts:
            s_req = max(vx, -vx/2 + vy*sqrt3/2, -vx/2 - vy*sqrt3/2, -vx, vx/2 - vy*sqrt3/2, vx/2 + vy*sqrt3/2)
            if s_req > min_S:
                min_S = s_req
                
    # Add a tiny epsilon to ensure strict inequality for the evaluator's 1e-6 tolerance
    # The evaluator checks: projected intervals are apart by MORE than 1e-6.
    # And cross product >= -1e-6.
    # So we need a small margin.
    S_final = min_S + 0.01
    
    # Construct output
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = opt_centers[i, 0]
        inner_hex_data[i, 1] = opt_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, S_final

# EVOLVE-BLOCK-END
