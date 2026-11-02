"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, rot_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side length, and rotation in degrees."""
    rot = np.radians(rot_deg)
    # Vertices of a regular hexagon of side length s, centered at origin, flat-top orientation
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + rot
    # For a regular hexagon with side length s, the distance from center to vertex is s
    # (since it's a regular hexagon, side length = circumradius)
    verts = side * np.column_stack([np.cos(angles), np.sin(angles)])
    return verts + np.array([cx, cy])

def _min_clearance_to_outer_boundary(inner_verts, outer_center, outer_side, outer_rot_deg):
    """
    Compute the minimum clearance (distance) between any inner hexagon vertex and the outer hexagon boundary.
    Positive means the vertex is inside the outer hexagon with that much clearance.
    Negative means the vertex is outside (penetration).
    
    For a regular hexagon, the distance from a point to the boundary can be computed by checking
    the distance to each of the 6 sides.
    """
    oc = np.array(outer_center)
    # Outer hexagon vertices
    outer_verts = _hex_vertices(0, 0, outer_side, outer_rot_deg) + oc
    
    # The 6 edges of the outer hexagon
    min_dist = np.inf
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        # Distance from each inner vertex to this edge segment
        for v in inner_verts:
            # Compute distance from point v to line segment p1-p2
            # But for a convex polygon, if the point is inside, the distance to the boundary
            # is the minimum over all edges of the perpendicular distance to the line containing the edge
            # (clamped to the segment endpoints).
            # Since we're dealing with a convex hexagon and points should be inside,
            # the perpendicular distance to the line should suffice if the projection falls within the segment.
            # For safety, let's compute proper point-to-segment distance.
            edge_vec = p2 - p1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq < 1e-12:
                d = np.linalg.norm(v - p1)
            else:
                t = np.dot(v - p1, edge_vec) / edge_len_sq
                t = np.clip(t, 0, 1)
                proj = p1 + t * edge_vec
                d = np.linalg.norm(v - proj)
            min_dist = min(min_dist, d)
    
    # But we need the SIGNED distance: positive if inside, negative if outside.
    # For a convex polygon, a point is inside if it's on the same side of each edge as the interior.
    # Let's determine if the point is inside the outer hexagon and compute signed distance.
    
    # Actually, let's use a simpler approach: compute the minimum distance to the boundary,
    # and determine the sign based on whether the point is inside.
    
    # Check if all inner vertices are inside the outer hexagon
    # For a convex polygon, point-in-polygon test
    inside = True
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        # Cross product to check side
        edge = p2 - p1
        for v in inner_verts:
            vec = v - p1
            cross = edge[0] * vec[1] - edge[1] * vec[0]
            # For a CCW polygon, interior is to the left, so cross should be >= 0
            # Let's check the orientation of the outer hexagon
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            break
    
    if inside:
        return min_dist
    else:
        return -min_dist

def _inner_hexagons_disjoint(inner_positions):
    """Check if the inner hexagons are disjoint (no overlap)."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            ci = inner_positions[i][:2]
            cj = inner_positions[j][:2]
            ri = inner_positions[i][2]
            rj = inner_positions[j][2]
            
            vi = _hex_vertices(ci[0], ci[1], 1.0, ri)
            vj = _hex_vertices(cj[0], cj[1], 1.0, rj)
            
            # Check if any vertex of one is inside the other, or if edges intersect
            # Simple check: min distance between vertices of different hexagons should be > 0
            # But this is expensive. Let's use a simpler heuristic:
            # If the centers are too close, they likely overlap.
            # For unit hexagons, the minimum distance between centers for non-overlapping is 2*sqrt(3)/2 = sqrt(3) ≈ 1.732
            # Actually, for two unit hexagons, the minimum center-to-center distance for non-overlap depends on orientation.
            # A safe lower bound is 2 (diameter of inscribed circle * 2 / 2 = sqrt(3) for inradius, so 2*inradius = sqrt(3))
            # Let's just check if any vertex of one hex is inside the other.
            
            # Point-in-hexagon test for each vertex of i against hexagon j
            # This is complex. Let's use a simpler distance-based check.
            # The minimum distance between any two vertices of different hexagons should be > 0.
            # But even that's not sufficient (edges could cross without vertices inside).
            
            # For now, let's use a conservative check: center distance > 2*sqrt(3) guarantees no overlap?
            # No, that's too conservative. The inradius of a unit hexagon is sqrt(3)/2 ≈ 0.866.
            # So if center distance > 2 * 0.866 = 1.732, they don't overlap.
            # But they could overlap even if center distance is slightly less than this.
            
            # Let's do a proper check: for each pair, compute the minimum distance between the two hexagons.
            # If min distance <= 0, they overlap.
            
            min_dist = np.inf
            for vi_point in vi:
                for vj_point in vj:
                    d = np.linalg.norm(vi_point - vj_point)
                    min_dist = min(min_dist, d)
            
            # Also check if any vertex of i is inside j, or vice versa
            # For simplicity, if min vertex-to-vertex distance is very small, they likely overlap
            if min_dist < 1e-6:
                return False
            
            # More thorough: check if any vertex of one is inside the other
            # Point in convex polygon
            for v in vi:
                if _point_in_hex(v, cj, 1.0, rj):
                    return False
            for v in vj:
                if _point_in_hex(v, ci, 1.0, ri):
                    return False
    
    return True

def _point_in_hex(point, center, side, rot_deg):
    """Check if a point is inside a regular hexagon."""
    verts = _hex_vertices(center[0], center[1], side, rot_deg)
    # Point in convex polygon test
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        vec = point - p1
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if cross < -1e-9:
            return False
    return True

def _objective(params, N):
    """
    Objective function to minimize.
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_x, outer_y, outer_side, outer_rot]
    We want to minimize outer_side, subject to:
    1. All inner hexagons are inside the outer hexagon (with clearance >= 0)
    2. Inner hexagons are disjoint
    
    We'll use a penalty method: minimize outer_side + penalty * (violation)
    """
    n = N
    inner_params = params[:3 * n]
    outer_params = params[3 * n:]
    outer_x, outer_y, outer_side, outer_rot = outer_params
    
    if outer_side < 0:
        return 1e10
    
    # Parse inner positions
    inner_positions = []
    for i in range(n):
        x = inner_params[3 * i]
        y = inner_params[3 * i + 1]
        a = inner_params[3 * i + 2]
        inner_positions.append([x, y, a])
    
    # Compute all inner vertices
    all_inner_verts = []
    for pos in inner_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, pos[2])
        all_inner_verts.append(verts)
    
    # Check clearance to outer boundary
    total_clearance_violation = 0.0
    for verts in all_inner_verts:
        for v in verts:
            # Distance from v to outer hexagon boundary
            # Compute signed distance
            oc = np.array([outer_x, outer_y])
            outer_verts = _hex_vertices(0, 0, outer_side, outer_rot) + oc
            
            # For each edge of the outer hexagon, compute distance
            min_dist = np.inf
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                edge_vec = p2 - p1
                edge_len_sq = np.dot(edge_vec, edge_vec)
                if edge_len_sq < 1e-12:
                    d = np.linalg.norm(v - p1)
                else:
                    t = np.dot(v - p1, edge_vec) / edge_len_sq
                    t = np.clip(t, 0, 1)
                    proj = p1 + t * edge_vec
                    d = np.linalg.norm(v - proj)
                min_dist = min(min_dist, d)
            
            # Determine if v is inside
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                edge = p2 - p1
                vec = v - p1
                cross = edge[0] * vec[1] - edge[1] * vec[0]
                if cross < -1e-9:
                    inside = False
                    break
            
            if not inside:
                total_clearance_violation += min_dist
            else:
                # If inside, no violation for this vertex
                pass
    
    # Check disjointness
    overlap_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            vi = all_inner_verts[i]
            vj = all_inner_verts[j]
            
            # Check vertex-to-vertex distances
            for v1 in vi:
                for v2 in vj:
                    d = np.linalg.norm(v1 - v2)
                    if d < 1e-4:
                        overlap_violation += (1e-4 - d)
            
            # Check if any vertex of i is inside j
            for v in vi:
                if _point_in_hex(v, inner_positions[j][:2], 1.0, inner_positions[j][2]):
                    overlap_violation += 1.0
            for v in vj:
                if _point_in_hex(v, inner_positions[i][:2], 1.0, inner_positions[i][2]):
                    overlap_violation += 1.0
    
    penalty = 100.0 * (total_clearance_violation + overlap_violation)
    return outer_side + penalty

def construct():
    """
    Find the optimal packing of N unit hexagons in a regular hexagon.
    Returns: (inner, center, side, angle)
    inner: list of [x, y, angle_degrees] for each inner hexagon
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation angle of outer hexagon in degrees
    """
    n = N
    
    # Initial guess: place hexagons in a hexagonal grid pattern
    # For N=11, we can use a 3-4-3 or similar arrangement
    # Let's start with a simple linear arrangement and let the optimizer improve it
    
    # Better initial guess: hexagonal close packing
    # For N hexagons, arrange in rows
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Or for N=11: rows of 3,4,4 or 2,3,3,3, etc.
    
    # Let's use a simple initial configuration
    initial_params = []
    
    # Place hexagons in a compact arrangement
    # Hexagonal grid spacing: sqrt(3) for horizontal, 1.5 for vertical (offset rows)
    dx = np.sqrt(3)
    dy = 1.5
    
    positions = []
    # Try a few different initial arrangements
    # Arrangement 1: 3-4-3
    row_sizes = [3, 4, 3]
    for r, size in enumerate(row_sizes):
        for c in range(size):
            x = c * dx - (size - 1) * dx / 2
            y = r * dy
            if r % 2 == 1:
                x += dx / 2
            positions.append([x, y, 0.0])
    
    # If we have more or fewer than N, adjust
    while len(positions) < n:
        positions.append([0.0, 0.0, 0.0])
    positions = positions[:n]
    
    for pos in positions:
        initial_params.extend([pos[0], pos[1], pos[2]])
    
    # Outer hexagon initial guess
    # Estimate required size
    outer_side_initial = 5.0
    initial_params.extend([0.0, 0.0, outer_side_initial, 0.0])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((-180, 180))  # angle
    bounds.append((-10, 10))  # outer_x
    bounds.append((-10, 10))  # outer_y
    bounds.append((1.0, 20.0))  # outer_side
    bounds.append((-180, 180))  # outer_rot
    
    # Try multiple starts
    best_result = None
    best_obj = np.inf
    
    n_starts = 5
    for start in range(n_starts):
        # Perturb initial parameters
        if start == 0:
            x0 = np.array(initial_params)
        else:
            x0 = np.array(initial_params) + np.random.uniform(-0.5, 0.5, size=len(initial_params))
            # Ensure outer_side is positive
            x0[3 * n + 2] = max(1.0, x0[3 * n + 2])
        
        result = minimize(
            _objective,
            x0,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Parse result
    x = best_result
    inner = []
    for i in range(n):
        inner.append([x[3 * i], x[3 * i + 1], x[3 * i + 2]])
    
    outer_x = x[3 * n]
    outer_y = x[3 * n + 1]
    outer_side = x[3 * n + 2]
    outer_rot = x[3 * n + 3]
    
    return inner, (outer_x, outer_y), outer_side, outer_rot
# EVOLVE-BLOCK-END
