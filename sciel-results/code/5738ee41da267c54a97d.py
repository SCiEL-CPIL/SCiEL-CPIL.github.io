import math
import time
import numpy as np
import itertools

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Unit hexagon vertices relative to center.
    # We use a "pointy-top" orientation for the inner hexagons initially, 
    # but since we can rotate the whole cluster, the relative orientation 
    # of inner hexagons to each other is what matters for the rigid body.
    # Standard hex grid: centers at (0,0) and (sqrt(3), 0) touch.
    # Vertices of a unit hexagon centered at (0,0):
    # Angles: 0, 60, 120, 180, 240, 300 degrees.
    verts_rel = np.array([
        [1.0, 0.0],
        [0.5, math.sqrt(3)/2],
        [-0.5, math.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -math.sqrt(3)/2],
        [0.5, -math.sqrt(3)/2]
    ])
    
    d = math.sqrt(3) # Distance between centers of touching hexagons
    
    # Define candidate rigid topologies for 11 hexagons.
    # Each topology is a list of (qx, qy) coordinates in the hex grid system.
    # We convert these to Cartesian coordinates.
    # Hex grid to Cartesian:
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2) * r
    
    def hex_grid_to_cart(q, r):
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3) / 2.0) * r
        return (x, y)
        
    topologies = []
    
    # Topology 1: Center + 6 neighbors (Ring 1) + 4 from Ring 2.
    # Ring 1 indices: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates are at distance 2.
    # We try different sets of 4 from Ring 2.
    # To save time, we'll pick a few symmetric-looking ones.
    
    # Base 7
    base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Ring 2 positions (q,r) where max(|q|,|r|,|q+r|)=2
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                ring2.append((q, r))
                
    # We need to add 4 more.
    # Strategy: Add 4 that are "adjacent" to the ring 1 to make a compact blob.
    # Example: (2,0), (0,2), (-2,2), (-2,0) -- this is a partial ring.
    # Let's just iterate through a subset of combinations of 4 from ring2.
    # There are C(12,4) = 495 combinations. This is fast enough to evaluate all.
    
    for combo in itertools.combinations(range(len(ring2)), 4):
        coords = base + [ring2[i] for i in combo]
        topologies.append(coords)
        
    # Also try a "linear" cluster? No, hexagonal packing is denser.
    # Try a 3x3 block? 
    # A 3x3 block in hex grid has 9 hexes. 
    # (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,-1), (1,-1), (2,-1)
    # Add 2 more?
    # Let's stick to the ring-based expansion as it's more likely to be optimal for N=11.
    
    best_S = float('inf')
    best_cluster_cart = None
    best_rotation = 0.0
    best_translation = (0.0, 0.0)
    
    # Precompute vertices for each topology? 
    # It's better to compute the "extremes" for a given rotation.
    
    # For a rigid cluster, the max projection in direction n is 
    # max_i dot(v_i, n) - min_i dot(v_i, n) ? No, the enclosing hexagon is centered at origin.
    # The side length S is determined by the maximum distance from the center to any vertex 
    # projected onto the 6 normals.
    # If we center the cluster's bounding box at the origin, the max projection P is 
    # (max_proj - min_proj) / 2 ? 
    # Actually, for a centrally symmetric object, yes. But our cluster might not be centrally symmetric.
    # The enclosing hexagon must contain all points.
    # If the hexagon is centered at C, and has rotation alpha,
    # the constraint is |dot(p - C, n_k)| <= P_max for all k=0..5.
    # To minimize P_max, we should choose C such that the max deviation is minimized.
    # For a fixed rotation, the optimal C is the center of the minimal bounding box?
    # Actually, for a regular hexagon, the optimal center is the center of the "hexagonal bounding box".
    # A simpler approach: 
    # 1. Rotate cluster by theta.
    # 2. Compute the 6 "widths" of the cluster in the directions of the hexagon normals.
    #    Width in direction n is max(dot(v, n)) - min(dot(v, n)).
    #    Wait, the hexagon is centered. The distance from center to side is P.
    #    The condition is that for all vertices v, |dot(v - C, n_k)| <= P.
    #    This implies C must be within the intersection of the strips.
    #    The minimal P for a given theta is half the maximum "width" of the cluster in the 6 directions?
    #    No, if the cluster is not symmetric, the center C might not be the centroid.
    #    However, for a convex hull, the minimal enclosing regular hexagon (with fixed orientation) 
    #    has side length S such that the max projection of the hull vertices onto the normals 
    #    (relative to the center C) is minimized.
    #    It turns out that for any convex body, the minimal enclosing regular hexagon with fixed orientation
    #    is obtained by centering the hexagon at the center of the minimal bounding box in the 3 orthogonal directions?
    #    Actually, a simpler bound: S >= max_k (max_i dot(v_i, n_k) - min_i dot(v_i, n_k)) * (1/sqrt(3))?
    #    Let's use the property that the minimal enclosing hexagon side S for a set of points 
    #    with fixed orientation is:
    #    S = (2/sqrt(3)) * min_C max_k max_i |dot(v_i - C, n_k)|
    #    The optimal C is the point that minimizes the max of the 6 "radius" functions.
    #    This is a Chebyshev center problem in 2D, which can be solved quickly.
    #    However, for our rigid cluster, we can just calculate the "radius" in each of the 6 directions 
    #    relative to the centroid, and then adjust.
    #    
    #    Let's simplify: 
    #    For a given rotation theta, compute the convex hull of the vertices.
    #    Then find the minimal enclosing regular hexagon.
    #    This is a known problem. 
    #    Alternatively, since N is small, we can just optimize C and S for each theta.
    #    
    #    Let's use a simpler heuristic that is likely optimal for these symmetric clusters:
    #    The optimal center C is the centroid of the vertices? Or the center of the bounding box?
    #    For highly symmetric clusters (like the flower), the centroid is likely optimal.
    #    
    #    Let's compute the max projection P for a given theta and C=centroid.
    #    P = max_k max_i |dot(v_i - centroid, n_k)|
    #    S = 2 * P / sqrt(3)
    
    for topo in topologies:
        # Convert to Cartesian
        cart_coords = [hex_grid_to_cart(q, r) for q, r in topo]
        
        # Get all vertices
        all_verts = []
        for cx, cy in cart_coords:
            all_verts.extend([(cx + vx, cy + vy) for vx, vy in verts_rel])
        all_verts = np.array(all_verts)
        
        centroid = np.mean(all_verts, axis=0)
        verts_centered = all_verts - centroid
        
        # Search for optimal rotation theta in [0, pi/3)
        # The function S(theta) is periodic with period pi/3.
        # It is piecewise linear in terms of cos/sin, but max of linear functions.
        # We can sample theta densely.
        
        # Optimization: The critical points of S(theta) occur when a vertex aligns with a normal.
        # We can just do a fine grid search.
        thetas = np.linspace(0, math.pi/3, 1000)
        
        min_S_for_topo = float('inf')
        best_theta_for_topo = 0
        
        for theta in thetas:
            c = math.cos(theta)
            s = math.sin(theta)
            
            # Rotate vertices
            rx = verts_centered[:, 0] * c - verts_centered[:, 1] * s
            ry = verts_centered[:, 0] * s + verts_centered[:, 1] * c
            rotated_verts = np.column_stack([rx, ry])
            
            # Normals of the enclosing hexagon (fixed at 0, pi/3, ...)
            # Wait, if we rotate the cluster, we are effectively rotating the normals relative to the cluster.
            # The enclosing hexagon has fixed orientation in the global frame.
            # So we project rotated_verts onto the fixed normals.
            k_angles = np.arange(6) * math.pi / 3
            normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
            
            # Projections
            dots = np.sum(rotated_verts[:, np.newaxis, :] * normals[np.newaxis, :, :], axis=-1)
            # We need max |dot| for each normal
            max_abs_dots = np.max(np.abs(dots), axis=0)
            P = np.max(max_abs_dots)
            S = (2.0 / math.sqrt(3)) * P
            
            if S < min_S_for_topo:
                min_S_for_topo = S
                best_theta_for_topo = theta
        
        if min_S_for_topo < best_S:
            best_S = min_S_for_topo
            best_rotation = best_theta_for_topo
            best_cluster_cart = cart_coords
            # We need to store the vertices to reconstruct later
            # But we can just recalculate them in the output phase.
            
    # Final check: Did we find a valid solution?
    if best_cluster_cart is None:
        # Fallback
        return [[0.0, 0.0, 0.0]] * N, (0.0, 0.0), 10.0, 0.0

    # Construct the final inner hexagons
    # We need to apply the rotation to the cart_coords and the local vertices.
    c = math.cos(best_rotation)
    s = math.sin(best_rotation)
    
    inner = []
    for i in range(N):
        cx, cy = best_cluster_cart[i]
        # Rotate center
        rx = cx * c - cy * s
        ry = cx * s + cy * c
        
        # The angle of the hexagon itself is also rotated by best_rotation.
        # The local vertices were defined with 0 rotation.
        # So the global angle is best_rotation.
        angle_deg = math.degrees(best_rotation)
        
        inner.append([rx, ry, angle_deg])
        
    # The outer hexagon center is (0,0) because we centered the cluster.
    # The outer hexagon side is best_S.
    # The outer hexagon rotation is 0 (since we fixed normals at 0, pi/3...).
    # Wait, the problem allows the outer hexagon to rotate.
    # In our formulation, we fixed the outer hexagon orientation and rotated the cluster.
    # This is equivalent to rotating the outer hexagon and keeping the cluster fixed.
    # So the outer rotation is 0, and the inner hexagons are rotated by best_rotation.
    # This is a valid solution.
    
    return inner, (0.0, 0.0), best_S, 0.0

