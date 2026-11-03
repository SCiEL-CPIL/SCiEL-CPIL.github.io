"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 120.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # In local frame, check if point is within the hexagon
    # The hexagon has width 2*side (horizontal) and height sqrt(3)*side (vertical)
    # For a regular hexagon with vertices at angles 0,60,...:
    # |x| <= side and |y| <= sqrt(3)/2 * side and |y| <= -sqrt(3)*|x| + sqrt(3)*side
    # Actually let's use the half-plane method
    # Vertices in local frame (side=1): (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # Edges define half-planes. For each edge, the inward normal points toward center.
    # Edge from v[i] to v[i+1]: normal = rotate edge by 90 deg inward
    s = side
    sqrt3 = np.sqrt(3)
    # The six half-plane inequalities for a unit hexagon centered at origin:
    # 1. x <= s
    # 2. -x <= s  => x >= -s
    # 3. sqrt(3)*y <= sqrt(3)*s  => y <= s
    # Wait, let me think more carefully.
    # Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # Edge 0: (s,0) to (s/2, s*sqrt(3)/2): direction = (-s/2, s*sqrt(3)/2), normal inward = (s*sqrt(3)/2, s/2) normalized -> (sqrt(3), 1)/2
    #   Inequality: sqrt(3)*x + y <= sqrt(3)*s
    # Edge 1: (s/2, s*sqrt(3)/2) to (-s/2, s*sqrt(3)/2): direction = (-s, 0), normal inward = (0, -s) -> (0,-1)
    #   Inequality: -y <= -s*sqrt(3)/2  => y >= -s*sqrt(3)/2... wait that's not right
    # Let me just check all 6 edges properly.
    # For edge from v_i to v_{i+1}, the inward normal n satisfies n·(p - v_i) <= 0 for all p inside.
    # Simpler: use the fact that a point is inside a convex polygon if it's on the correct side of each edge.
    
    verts_local = []
    for i in range(6):
        theta = i * np.pi / 3
        verts_local.append([s * np.cos(theta), s * np.sin(theta)])
    verts_local = np.array(verts_local)
    
    for i in range(6):
        v1 = verts_local[i]
        v2 = verts_local[(i + 1) % 6]
        edge = v2 - v1
        # Inward normal (rotate edge 90 deg clockwise: (dy, -dx))
        nx, ny = edge[1], -edge[0]
        # Check: n·(p - v1) <= 0
        if nx * (lx - v1[0]) + ny * (ly - v1[1]) > 1e-10:
            return False
    return True

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            # Normal to edge
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    """Find minimum outer side length such that all inner hexagons fit inside the outer hexagon."""
    cx, cy = outer_center
    # For each inner hexagon, find the maximum "extent" in terms of required outer side
    # The outer hexagon contains a point (px, py) iff the point is inside it.
    # For a given outer hexagon with center (cx,cy), side s, angle a, a point is inside iff
    # it satisfies all 6 half-plane constraints.
    # We need the smallest s such that all vertices of all inner hexagons are inside.
    
    # Collect all vertices of inner hexagons
    all_verts = []
    for i in range(len(inner_positions)):
        x, y = inner_positions[i]
        ang = inner_angles[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Binary search on outer side length
    lo, hi = 1.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        # Check if all vertices are inside outer hexagon with side mid
        outer_verts = _hex_vertices(cx, cy, mid, outer_angle)
        # For each inner vertex, check if inside outer hexagon
        ok = True
        for v in all_verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, mid, outer_angle):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi

def _distance_to_hex_edge(px, py, cx, cy, side, angle_deg):
    """Return signed distance from point to hexagon boundary (positive inside)."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    s = side
    sqrt3 = np.sqrt(3)
    
    # Compute the 6 half-plane distances
    # Edge normals and offsets for unit hexagon centered at origin
    # Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    # The 6 supporting lines:
    # 1. x = s  -> distance = s - lx
    # 2. -x = s -> distance = s + lx
    # 3. sqrt(3)*x + y = sqrt(3)*s -> distance = (sqrt(3)*s - (sqrt(3)*lx + ly)) / 2
    # 4. -sqrt(3)*x + y = sqrt(3)*s -> distance = (sqrt(3)*s - (-sqrt(3)*lx + ly)) / 2
    # 5. -sqrt(3)*x - y = sqrt(3)*s -> distance = (sqrt(3)*s - (-sqrt(3)*lx - ly)) / 2
    # 6. sqrt(3)*x - y = sqrt(3)*s -> distance = (sqrt(3)*s - (sqrt(3)*lx - ly)) / 2
    
    d1 = s - lx
    d2 = s + lx
    d3 = (sqrt3 * s - (sqrt3 * lx + ly)) / 2.0
    d4 = (sqrt3 * s - (-sqrt3 * lx + ly)) / 2.0
    d5 = (sqrt3 * s - (-sqrt3 * lx - ly)) / 2.0
    d6 = (sqrt3 * s - (sqrt3 * lx - ly)) / 2.0
    
    return min(d1, d2, d3, d4, d5, d6)

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check no overlaps
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    # Check all inside outer hexagon
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _initial_packing():
    """Create initial hexagonal grid packing."""
    positions = []
    angles = []
    # Hexagonal close packing: offset rows
    # For unit hexagons, horizontal spacing = 2, vertical spacing = sqrt(3)
    # Offset every other row by 1
    sqrt3 = np.sqrt(3)
    rows = 3
    for r in range(rows):
        y = (r - rows // 2) * sqrt3
        if r % 2 == 0:
            xs = range(-(rows // 2), rows - rows // 2 + 1)
        else:
            xs = range(-(rows // 2), rows - rows // 2)
        for x_idx in xs:
            x = (x_idx - (rows // 2)) * 2.0
            if r % 2 == 1:
                x += 1.0
            positions.append([x, y])
            angles.append(0.0)
    # Trim to N hexagons
    positions = positions[:N]
    angles = angles[:N]
    return np.array(positions), np.array(angles)

def _smooth_packing_objective(params, inner_angles, outer_center, outer_angle, beta=50.0):
    """
    Smooth objective: minimize the soft-max of distances from all inner hexagon vertices
    to the outer hexagon boundary.
    params: flattened array of [x_0, y_0, x_1, y_1, ..., x_{n-1}, y_{n-1}]
    Returns a scalar that is approximately the minimum required outer side when all
    vertices are inside.
    """
    n = len(inner_angles)
    pos = params.reshape(n, 2)
    cx, cy = outer_center
    
    # Collect all vertex distances to outer hexagon boundary
    # We want to maximize the minimum distance (or minimize the maximum "penetration")
    # Actually, we want the outer hexagon to contain all vertices.
    # The required outer side is determined by the vertex that is "furthest out".
    # For a given outer hexagon orientation, the required side length is related to
    # the support function.
    
    # Simpler approach: for each vertex, compute what outer side length would be needed
    # if that vertex were on the boundary. Then take the max.
    # But this is complex. Instead, use a penalty-based approach.
    
    # Alternative: minimize the maximum of (distance_to_boundary < 0) penalties
    # Let's use: for each vertex, compute d = distance to boundary (negative if outside)
    # We want all d >= 0. Minimize max(-d) which is the maximum penetration.
    
    all_dists = []
    for i in range(n):
        x, y = pos[i]
        ang = inner_angles[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        for v in verts:
            # Distance to outer hexagon boundary
            d = _distance_to_hex_edge(v[0], v[1], cx, cy, 1.0, outer_angle)
            # If d < 0, the point is outside a unit hexagon
            # The required outer side to contain this point is related to 1/d if d > 0
            # Actually, the distance scales linearly with side length
            # If point is at distance d from boundary of unit hexagon, it needs side = 1/d? No.
            # If we scale the hexagon by factor s, distances scale by s.
            # A point inside a unit hexagon at distance d from boundary is inside an s-hexagon
            # at distance s*d from boundary.
            # We need s*d >= 0, i.e., the point must be inside the unit hexagon (d >= 0).
            # If d < 0, we need to increase s until the point is inside.
            # The required s is such that the point is on the boundary: s = 1/(-d) * something?
            # Actually, if a point is at signed distance d from the boundary of a unit hexagon
            # (positive inside), then for a hexagon of side s, the signed distance is s*d.
            # We need s*d >= 0 for all points. If d < 0 for some point, no finite s works
            # unless we move the point.
            # So the constraint is: all points must be inside the unit hexagon (d >= 0).
            # Then the minimum s is 1.0? That's not right either.
            
            # Let me reconsider. The outer hexagon has side length S.
            # A point is inside if it satisfies the half-plane constraints for side S.
            # The distance to boundary scales with S.
            # If a point is at distance d_unit from the boundary of a unit hexagon (centered at origin),
            # then for a hexagon of side S, the distance is S * d_unit.
            # The point is inside iff S * d_unit >= 0 for all 6 half-planes.
            # This means d_unit >= 0 for all 6 half-planes, i.e., the point is inside the unit hexagon.
            # So if all points are inside the unit hexagon, any S >= 1 works? No, that's wrong.
            
            # I think I'm confusing myself. Let me just use the original binary search approach
            # but make it differentiable by using a smooth max.
            pass
    
    # This approach is getting complicated. Let me use a different smooth objective.
    # Objective: minimize the maximum over all vertices of the "required outer side" contribution.
    # For a vertex at position (vx, vy) relative to outer center, the required outer side
    # can be computed analytically.
    
    # For a regular hexagon centered at origin with side S and angle a, a point (px, py)
    # is inside iff for all 6 edges, the point is on the correct side.
    # The required S for a single point is the maximum over the 6 constraints.
    
    # This is still complex. Let me just use the original approach but with a smoother
    # local search.
    return 0.0  # Placeholder

def _min_side_from_params(params, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side from parameter vector."""
    n = len(inner_angles)
    pos = params.reshape(n, 2)
    return _min_outer_side(pos, inner_angles, outer_center, outer_angle)

def construct():
    global START_TIME, DEADLINE
    
    # Initial packing
    pos, ang = _initial_packing()
    n = len(pos)
    
    # Outer hexagon center and angle (start at origin, no rotation)
    outer_cx, outer_cy = 0.0, 0.0
    outer_angle = 0.0
    
    # Initial side estimate
    current_side = _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle)
    
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_ocx, best_ocy = outer_cx, outer_cy
    best_oor = outer_angle
    
    # Convert to flat parameter vector for scipy optimization
    best_params = np.concatenate([best_pos.flatten(), best_ang])
    
    # Use scipy.optimize.minimize with Nelder-Mead or Powell
    # Objective: minimize outer side length
    def objective(params):
        if time.time() > DEADLINE - 10.0:
            return best_side + 1000  # Penalty if we're running out of time
        pos = params[:2*n].reshape(n, 2)
        ang = params[2*n:]
        try:
            side = _min_outer_side(pos, ang, (outer_cx, outer_cy), outer_angle)
            # Check for overlaps (penalty if overlapping)
            verts_list = []
            for i in range(n):
                verts = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
                verts_list.append(verts)
            for i in range(n):
                for j in range(i+1, n):
                    if _hexagon_overlap(verts_list[i], verts_list[j]):
                        return side + 100.0  # Penalty for overlap
            return side
        except Exception:
            return best_side + 1000
    
    # Run scipy optimization
    if time.time() < DEADLINE - 30.0:
        try:
            result = minimize(
                objective,
                best_params.copy(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if result.fun < best_side:
                best_params = result.x
                best_pos = best_params[:2*n].reshape(n, 2)
                best_ang = best_params[2*n:]
                best_side = result.fun
                # Remove penalty if any
                if best_side > 50.0:
                    # Probably had overlap penalty, recompute
                    best_side = _min_outer_side(best_pos, best_ang, (outer_cx, outer_cy), outer_angle)
        except Exception:
            pass
    
    # Final local search polish
    step = 0.05
    improved = True
    while improved and time.time() < DEADLINE - 5.0:
        improved = False
        for i in range(n):
            if time.time() > DEADLINE - 5.0:
                break
            for dx in [-step, 0.0, step]:
                for dy in [-step, 0.0, step]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    new_pos = best_pos.copy()
                    new_pos[i][0] += dx
                    new_pos[i][1] += dy
                    
                    # Check validity
                    verts = _hex_vertices(new_pos[i][0], new_pos[i][1], 1.0, best_ang[i])
                    inside = True
                    for v in verts:
                        if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, best_side, outer_angle):
                            inside = False
                            break
                    if not inside:
                        continue
                    
                    overlap = False
                    for j in range(n):
                        if j == i:
                            continue
                        verts_j = _hex_vertices(best_pos[j][0], best_pos[j][1], 1.0, best_ang[j])
                        if _hexagon_overlap(verts, verts_j):
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    new_side = _min_outer_side(new_pos, best_ang, (outer_cx, outer_cy), outer_angle)
                    if new_side < best_side - 1e-8:
                        best_pos = new_pos
                        best_side = new_side
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.7
            if step < 0.001:
                break
    
    # Final verification
    if not _check_validity(
        [[best_pos[i][0], best_pos[i][1], best_ang[i]] for i in range(N)],
        (best_ocx, best_ocy), best_side, best_oor
    ):
        # Fallback to initial
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    return inner, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oor)

# EVOLVE-BLOCK-END
