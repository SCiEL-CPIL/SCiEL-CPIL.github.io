"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull

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
    
    # Local search: perturb positions and angles
    np.random.seed(42)
    step = 0.1
    iters = 0
    
    while time.time() < DEADLINE - 5.0:
        iters += 1
        
        # Try perturbing one hexagon at a time
        improved = False
        for i in range(n):
            if time.time() > DEADLINE - 5.0:
                break
            
            for dx in [-step, 0.0, step]:
                for dy in [-step, 0.0, step]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    new_pos = pos.copy()
                    new_pos[i][0] += dx
                    new_pos[i][1] += dy
                    
                    # Check validity quickly: just check if still inside outer hexagon
                    # (skip overlap check for speed during search, do full check at end)
                    verts = _hex_vertices(new_pos[i][0], new_pos[i][1], 1.0, ang[i])
                    inside = True
                    for v in verts:
                        if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, current_side, outer_angle):
                            inside = False
                            break
                    if not inside:
                        continue
                    
                    # Check overlap with others
                    overlap = False
                    for j in range(n):
                        if j == i:
                            continue
                        verts_j = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                        if _hexagon_overlap(verts, verts_j):
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    # Compute new required outer side
                    new_side = _min_outer_side(new_pos, ang, (outer_cx, outer_cy), outer_angle)
                    if new_side < current_side - 1e-8:
                        pos = new_pos
                        current_side = new_side
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try angle perturbations
            for i in range(n):
                if time.time() > DEADLINE - 5.0:
                    break
                for da in [-5.0, 5.0]:
                    new_ang = ang.copy()
                    new_ang[i] = ang[i] + da
                    new_side = _min_outer_side(pos, new_ang, (outer_cx, outer_cy), outer_angle)
                    if new_side < current_side - 1e-8:
                        # Check validity
                        verts = _hex_vertices(pos[i][0], pos[i][1], 1.0, new_ang[i])
                        inside = True
                        for v in verts:
                            if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, current_side, outer_angle):
                                inside = False
                                break
                        if not inside:
                            continue
                        overlap = False
                        for j in range(n):
                            if j == i:
                                continue
                            verts_j = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                            if _hexagon_overlap(verts, verts_j):
                                overlap = True
                                break
                        if not overlap:
                            ang = new_ang
                            current_side = new_side
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            # Decrease step
            step *= 0.8
            if step < 0.001:
                break
        
        # Update best
        if current_side < best_side:
            best_side = current_side
            best_pos = pos.copy()
            best_ang = ang.copy()
            best_ocx, best_ocy = outer_cx, outer_cy
            best_oor = outer_angle
    
    # Final verification
    if not _check_validity(
        [[best_pos[i][0], best_pos[i][1], best_ang[i]] for i in range(N)],
        (best_ocx, best_ocy), best_side, best_oor
    ):
        # Fallback to initial
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner_list = [[float(best_pos[i][0]), float(best_pos[i][1]), float(best_ang[i])] for i in range(N)]
    if not _check_validity(inner_list, (best_ocx, best_ocy), best_side, best_oor):
        sqrt3 = np.sqrt(3)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return inner_list, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oor)

# EVOLVE-BLOCK-END
