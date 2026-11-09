# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _hex_normals(a_deg):
    """12 edge normals (unit vectors) for a hexagon of rotation a_deg."""
    a = np.radians(a_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])

def _separation_ok(v1, v2, n1, n2):
    """Check if two hexagons are separated."""
    for normals in (n1, n2):
        for j in range(normals.shape[0]):
            n = normals[j]
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
    return False

def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)

def _point_in_hexagon(p, outer_verts):
    """Check if point p is inside (or on boundary of) the outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True

def _all_inner_in_outer(inner_verts_list, outer_verts):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True

def _compute_minimal_outer_hex(all_verts):
    """
    Compute the minimal enclosing regular hexagon for a set of points.
    Returns (center_x, center_y, angle_deg, side_length).
    """
    # Use convex hull to reduce points
    try:
        hull = ConvexHull(all_verts)
        hull_verts = all_verts[hull.vertices]
    except Exception:
        hull_verts = all_verts
    
    # If hull has fewer than 3 points, handle edge case
    if len(hull_verts) < 3:
        # Center at centroid, radius is max distance
        center = np.mean(hull_verts, axis=0) if len(hull_verts) > 0 else np.array([0.0, 0.0])
        if len(hull_verts) > 0:
            dists = np.linalg.norm(hull_verts - center, axis=1)
            r = np.max(dists)
            s = r / (np.sqrt(3) / 2) # Apothem = s*sqrt(3)/2, circumradius = s. 
            # Wait, for regular hexagon, circumradius = side length.
            # Apothem = s * sqrt(3) / 2.
            # Max distance from center to vertex of inner hex must be <= circumradius of outer hex?
            # No, the inner hex vertices must be inside the outer hex.
            # The farthest point from center in a regular hexagon is its vertex (distance s).
            # So s_outer >= max_dist / 1.0?
            # Actually, the constraint is that the point must be inside the polygon.
            # For a regular hexagon, the distance to the boundary varies between apothem and circumradius.
            # To be safe, we just need to ensure the points are inside.
            # Let's stick to the general optimization which handles this.
            return (center[0], center[1], 0.0, s)

    best_s = np.inf
    best_center = np.mean(hull_verts, axis=0)
    best_angle = 0.0
    
    # The optimal orientation of the enclosing hexagon is likely aligned with one of the hull edges or vertices.
    # We can sample angles.
    # Also, the center is likely the centroid of the hull or a point that balances the max projections.
    
    # Strategy: Iterate over angles. For each angle, find the optimal center that minimizes the max projection range.
    # For a fixed angle, the minimal enclosing hexagon centered at C has side s such that:
    # For each of 6 normals n_i: max(p @ n_i) - C @ n_i <= apothem and C @ n_i - min(p @ n_i) <= apothem
    # So 2 * apothem >= max_i (max(p @ n_i) - min(p @ n_i))
    # This implies apothem >= 0.5 * max_i (range_i)
    # But the center C affects the projections.
    # Actually, if we fix the angle, the minimal apothem is determined by the width of the point set in the 6 normal directions.
    # The width in direction n is max(p @ n) - min(p @ n).
    # The center must be chosen such that the projections fit in [C @ n - apothem, C @ n + apothem].
    # This is possible iff C @ n is between min(p @ n) + apothem and max(p @ n) - apothem.
    # So we need an intersection of intervals for C @ n_i for all i.
    # This is a linear programming problem or can be solved by checking if the intersection is non-empty.
    # To minimize s (and thus apothem), we find the smallest apothem such that the intersection is non-empty.
    # This is equivalent to finding the minimal width of the set in the hexagonal metric.
    
    # A simpler approach for "minimal enclosing regular hexagon":
    # The optimal hexagon will have at least 3 vertices of the inner set touching its edges.
    # We can use a rotating calipers-like approach or just grid search over angle and center.
    
    # Given the small number of points (<= 66), we can afford a finer search.
    # Let's assume the center is near the centroid.
    centroid = np.mean(hull_verts, axis=0)
    
    # Search over angle 0 to 60 degrees
    # And search over center in a small region around centroid
    # Or better: for each angle, the optimal center is the one that minimizes the max "excess" projection.
    # Let's just do a coarse-to-fine search over angle and center.
    
    # Actually, there is a known result: the minimal enclosing regular polygon can be found by rotating the polygon.
    # Let's try sampling angles.
    
    angles = np.linspace(0, 60, 121)
    for a_deg in angles:
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        # Projections of hull verts onto normals
        proj = hull_verts @ normals.T  # (M, 6)
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        # We need to find center C and apothem A such that:
        # C @ n_i - A <= min_proj_i
        # C @ n_i + A >= max_proj_i
        # => A >= C @ n_i - min_proj_i
        # => A >= max_proj_i - C @ n_i
        # So A >= max_i (max(C @ n_i - min_proj_i, max_proj_i - C @ n_i))
        # Let f(C) = max_i (max(C @ n_i - min_proj_i, max_proj_i - C @ n_i))
        # We want to minimize f(C). This is a convex function of C.
        # We can minimize it using scipy.optimize.minimize (Nelder-Mead) for each angle.
        # But doing this for 121 angles might be slow.
        
        # Alternative: The optimal C is the center of the minimal enclosing circle in the "hexagonal metric"?
        # Let's just use a simple heuristic: The center C that equalizes the max and min projections for the most constrained directions.
        # Or simply, let's try the centroid first.
        
        C = centroid
        c_proj = C @ normals.T
        # Required apothem
        A = np.max(np.maximum(c_proj - min_proj, max_proj - c_proj))
        s = A / (np.sqrt(3) / 2)
        
        # Try a few local optimizations for C
        def obj_c(c):
            cp = c @ normals.T
            return np.max(np.maximum(cp - min_proj, max_proj - c_proj))
            
        # Use a few iterations of gradient descent or Nelder-Mead for C
        # Since M is small, this is fast.
        from scipy.optimize import minimize
        res = minimize(obj_c, centroid, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < A:
            A = res.fun
            C = res.x
            s = A / (np.sqrt(3) / 2)
            
        if s < best_s:
            best_s = s
            best_center = C
            best_angle = a_deg
            
    return (best_center[0], best_center[1], best_angle, best_s)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial configuration: Hexagonal lattice
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more at distance 2*sqrt(3) in specific directions?
    # Let's just start with a tight cluster and let physics sort it out.
    
    np.random.seed(42)
    
    # 11 hexagons
    # Start with a rough hexagonal arrangement
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # Ring 1: 6 hexagons
    r1 = np.sqrt(3)
    for i in range(6):
        ang = np.deg2rad(60 * i)
        positions.append([r1 * np.cos(ang), r1 * np.sin(ang)])
        angles.append(0.0)
        
    # Ring 2: 4 hexagons to make 11 total
    # Place them in the gaps or further out
    r2 = 2 * np.sqrt(3)
    for i in range(4):
        ang = np.deg2rad(60 * i + 30) # Offset by 30 to fill gaps?
        positions.append([r2 * np.cos(ang), r2 * np.sin(ang)])
        angles.append(0.0)
        
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Physics simulation
    # We will simulate collisions between hexagons.
    # To simplify, we approximate each hexagon as a circle with radius = circumradius = 1.0?
    # No, that's too loose.
    # Approximate as circle with radius = inradius = sqrt(3)/2 ~ 0.866?
    # If we use inradius, they can interlock.
    # Let's use a distance-based repulsion that respects the hexagonal shape roughly.
    # Actually, a better approach: 
    # 1. Run a simple relaxation where we push overlapping hexagons apart.
    # 2. Overlap check: use the separation test.
    # 3. If not separated, calculate a correction vector.
    
    # For speed and simplicity, let's use a "soft" potential based on distance, 
    # but tuned so that the equilibrium distance corresponds to touching hexagons.
    # Min distance for touching hexagons (same orientation) is sqrt(3) (edge to edge) or 2 (vertex to vertex).
    # Average is around 1.732.
    # Let's target distance d_min = 1.732.
    
    d_min = np.sqrt(3)
    
    for step in range(500):
        if time.time() > deadline - 10:
            break
            
        forces = np.zeros_like(positions)
        
        for i in range(11):
            for j in range(i + 1, 11):
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r)
                if dist < 1e-6:
                    # Random push
                    forces[i] += np.random.randn(2) * 0.1
                    forces[j] -= np.random.randn(2) * 0.1
                    continue
                
                if dist < d_min:
                    # Repulsive force
                    # F = k * (d_min - dist)
                    k = 1.0
                    f_mag = k * (d_min - dist)
                    f_dir = r / dist
                    forces[i] -= f_dir * f_mag
                    forces[j] += f_dir * f_mag
                    
        # Update positions
        # Damping
        positions += forces * 0.1
        # Keep them roughly centered
        centroid = np.mean(positions, axis=0)
        positions -= centroid
        
    # After simulation, we have a configuration.
    # Now we need to ensure they are actually separated (no intersection).
    # The circle approximation might allow intersections.
    # We need to run a "collision resolution" loop using the exact hexagon separation test.
    
    # Build hexagons
    hexagons = []
    for i in range(11):
        hexagons.append((positions[i][0], positions[i][1], angles[i]))
        
    # Iterate to resolve collisions
    for iter in range(100):
        if time.time() > deadline - 5:
            break
            
        verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
        collision_found = False
        
        for i in range(11):
            for j in range(i + 1, 11):
                if not _hexagons_separated(verts_list[i], verts_list[j], hexagons[i][2], hexagons[j][2]):
                    # Collision!
                    collision_found = True
                    # Push apart
                    r = positions[j] - positions[i]
                    dist = np.linalg.norm(r)
                    if dist < 1e-6:
                        r = np.random.randn(2)
                        dist = np.linalg.norm(r)
                    f_dir = r / dist
                    # Push by a small amount
                    push = 0.01
                    positions[i] -= f_dir * push
                    positions[j] += f_dir * push
                    hexagons[i] = (positions[i][0], positions[i][1], angles[i])
                    hexagons[j] = (positions[j][0], positions[j][1], angles[j])
                    
        if not collision_found:
            break
            
    # Final check and adjustment
    # If still colliding, we might need to expand the outer hexagon, but that lowers the score.
    # We hope the simulation found a valid packing.
    
    # Re-center the whole cluster
    centroid = np.mean(positions, axis=0)
    positions -= centroid
    hexagons = [(positions[i][0], positions[i][1], angles[i]) for i in range(11)]
    
    inner_hex_data = np.array(hexagons)
    
    # Compute minimal enclosing hexagon
    all_verts = np.vstack([_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons])
    ox, oy, oa, s = _compute_minimal_outer_hex(all_verts)
    
    outer_hex_data = np.array([ox, oy, oa])
    
    # Verify containment
    outer_verts = _hex_vertices(ox, oy, s, oa)
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
    
    if not _all_inner_in_outer(verts_list, outer_verts):
        # Expand s slightly
        for _ in range(20):
            s *= 1.01
            outer_verts = _hex_vertices(ox, oy, s, oa)
            if _all_inner_in_outer(verts_list, outer_verts):
                break
                
    # Ensure s >= sqrt(11)
    s = max(s, np.sqrt(11.0))
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
