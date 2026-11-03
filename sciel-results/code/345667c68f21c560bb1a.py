"""Graph-theoretic rigid-cluster approach for hexagon packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 120.0

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_vertices_batch(positions, angles, side=1.0):
    """Vectorized vertices for N hexagons."""
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = side * np.cos(total_angles)
    offsets_y = side * np.sin(total_angles)
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    return np.column_stack([verts_x.ravel(), verts_y.ravel()])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s = side
    sqrt3 = np.sqrt(3)
    if lx > s + 1e-10 or lx < -s - 1e-10:
        return False
    if sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx + ly > sqrt3 * s + 1e-10:
        return False
    if -sqrt3 * lx - ly > sqrt3 * s + 1e-10:
        return False
    if sqrt3 * lx - ly > sqrt3 * s + 1e-10:
        return False
    return True

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    """Compute minimum outer side length S for all vertices to fit inside outer hexagon."""
    verts = np.asarray(all_verts, dtype=np.float64)
    n_pts = len(verts)
    if n_pts == 0:
        return 0.0
    
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = lx
    c2 = -lx
    c3 = lx + ly / sqrt3
    c4 = -lx + ly / sqrt3
    c5 = -lx - ly / sqrt3
    c6 = lx - ly / sqrt3
    
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """SAT overlap check for two convex hexagons."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
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

def _check_no_overlap(positions, angles, tol=1e-7):
    """Check no two hexagons overlap."""
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """Full validity check."""
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _hexagon_overlap(verts_list[i], verts_list[j]):
                return False
    
    for verts in verts_list:
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                return False
    return True

def _build_hex_cluster_positions():
    """
    Build 11 unit hexagons in a compact hexagonal cluster arrangement.
    For unit hexagons (side=1), the center-to-center distance for adjacent
    hexagons in a hex grid is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ≈ 3.464 for 
    pointy-top, or we use the standard hex grid spacing.
    
    For pointy-top hexagons (vertices at 0°, 60°, 120°, 180°, 240°, 300°):
    - Width (horizontal, vertex to vertex) = 2
    - Height (vertical, vertex to vertex) = sqrt(3)
    - For non-overlapping: horizontal spacing = sqrt(3), vertical spacing = 1.5
    - Odd rows offset by sqrt(3)/2
    
    Actually, let me reconsider. For a regular hexagon with side length s:
    - The distance from center to vertex (circumradius) = s
    - The distance from center to edge midpoint (apothem) = s * sqrt(3)/2
    
    For two adjacent unit hexagons to just touch, their centers must be 
    separated by 2 * apothem = sqrt(3) in the direction perpendicular to the shared edge.
    
    For a hex grid of pointy-top hexagons:
    - Horizontal spacing between columns: 2 * apothem = sqrt(3)
    - Vertical spacing between rows: 1.5 * side = 1.5
    - Odd rows offset horizontally by sqrt(3)/2
    """
    sqrt3 = np.sqrt(3)
    
    # Hex grid positions for 11 hexagons (compact cluster)
    # Row 0: 1 hex at (0, 0)
    # Row 1: 2 hexes at (-sqrt3/2, 1.5), (sqrt3/2, 1.5)
    # Row 2: 3 hexes at (-sqrt3, 3.0), (0, 3.0), (sqrt3, 3.0)
    # Row 3: 2 hexes at (-sqrt3/2, 4.5), (sqrt3/2, 4.5)
    # Row 4: 1 hex at (0, 6.0)
    # But that's 9. Let me add 2 more.
    
    # Alternative: use a more compact arrangement
    # Actually for 11, let's use:
    # Row 0: 3 hexes
    # Row 1: 3 hexes (offset)
    # Row 2: 3 hexes
    # Row 3: 2 hexes (offset)
    
    positions = []
    
    # Using pointy-top hex grid
    # For unit hexagons, adjacent centers are sqrt(3) apart horizontally
    # and 1.5 apart vertically, with odd rows offset by sqrt(3)/2
    
    # Let me use a standard hexagonal number arrangement
    # Center hex + 6 around it + 4 more = 11
    
    # Actually, let me just use a clean hex grid:
    # 4 rows: 3, 4, 3, 1 = 11
    # Or: 3, 3, 3, 2 = 11
    
    # Let's use the classic hex cluster:
    # Row 0 (y=0): x = -sqrt3, 0, sqrt3  (3 hexes)
    # Row 1 (y=1.5): x = -sqrt3/2, sqrt3/2  (2 hexes, offset)
    # Row 2 (y=3.0): x = -sqrt3, 0, sqrt3  (3 hexes)
    # Row 3 (y=4.5): x = -sqrt3/2, sqrt3/2  (2 hexes, offset)
    # Row 4 (y=6.0): x = 0  (1 hex)
    # Total: 3+2+3+2+1 = 11
    
    rows = [
        (0.0, [-sqrt3, 0, sqrt3]),
        (1.5, [-sqrt3/2, sqrt3/2]),
        (3.0, [-sqrt3, 0, sqrt3]),
        (4.5, [-sqrt3/2, sqrt3/2]),
        (6.0, [0]),
    ]
    
    for y, xs in rows:
        for x in xs:
            positions.append([x, y])
    
    positions = np.array(positions, dtype=np.float64)
    
    # Center the cluster
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    positions = positions - np.array([cx, cy])
    
    return positions

def _rotate_and_translate(positions, cluster_angle_deg, tx, ty):
    """Rotate positions by cluster_angle and translate by (tx, ty)."""
    a = np.radians(cluster_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    rotated = np.column_stack([
        ca * positions[:, 0] - sa * positions[:, 1],
        sa * positions[:, 0] + ca * positions[:, 1]
    ])
    return rotated + np.array([tx, ty])

def _objective_rigid_cluster(params, base_positions):
    """
    Objective: minimize the outer hexagon side length for a rigid cluster.
    params = [cluster_angle_deg, tx, ty]
    The outer hexagon is centered at origin with angle 0.
    """
    cluster_angle, tx, ty = params
    
    # Get actual positions
    positions = _rotate_and_translate(base_positions, cluster_angle, tx, ty)
    angles = np.zeros(N)  # All hexagons have same orientation as cluster rotation
    
    # Actually, if we rotate the cluster, each hexagon's angle also rotates
    angles = np.full(N, cluster_angle)
    
    # Compute all vertices
    all_verts = _hex_vertices_batch(positions, angles, side=1.0)
    
    # Outer hexagon centered at (0,0) with angle 0
    required_side = _min_side_vectorized(all_verts, 0.0, 0.0, 0.0)
    
    # Check for overlaps - if any overlap, add large penalty
    overlap_penalty = 0.0
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol=1e-8):
                # Compute penetration depth approximation using center distance
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Minimum distance for non-overlap is sqrt(3) for pointy-top
                min_dist = np.sqrt(3)
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2
    
    return required_side + 1000.0 * overlap_penalty

def _objective_with_outer_center(params, base_positions):
    """
    Extended objective: also optimize outer center.
    params = [cluster_angle_deg, tx, ty, outer_cx, outer_cy]
    """
    cluster_angle, tx, ty, ocx, ocy = params
    
    positions = _rotate_and_translate(base_positions, cluster_angle, tx, ty)
    angles = np.full(N, cluster_angle)
    
    all_verts = _hex_vertices_batch(positions, angles, side=1.0)
    
    required_side = _min_side_vectorized(all_verts, ocx, ocy, 0.0)
    
    # Overlap penalty
    overlap_penalty = 0.0
    verts_list = []
    for i in range(N):
        verts_list.append(_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol=1e-8):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = np.sqrt(3)
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2
    
    return required_side + 1000.0 * overlap_penalty

def _smooth_overlap_penalty(positions, angles):
    """Smooth penalty for near-overlaps using center distances."""
    penalty = 0.0
    min_dist = np.sqrt(3)  # Minimum center distance for unit hexagons
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist + 0.5:
                # Smooth quadratic penalty
                excess = max(0, min_dist - dist)
                penalty += excess ** 2
    return penalty

def construct():
    global START_TIME, DEADLINE
    
    # Build base cluster positions
    base_positions = _build_hex_cluster_positions()
    
    # Initial parameters: [cluster_angle, tx, ty]
    # Start with no rotation, centered at origin
    best_params = np.array([0.0, 0.0, 0.0])
    best_side = _objective_rigid_cluster(best_params, base_positions)
    
    # Try multiple initial rotations
    best_overall_params = best_params.copy()
    best_overall_side = best_side
    
    for init_angle in [0, 30, 60, 15, 45, 90, -30, -15]:
        if time.time() > DEADLINE - 20.0:
            break
        p = np.array([float(init_angle), 0.0, 0.0])
        s = _objective_rigid_cluster(p, base_positions)
        if s < best_overall_side:
            best_overall_side = s
            best_overall_params = p.copy()
    
    best_params = best_overall_params.copy()
    best_side = best_overall_side
    
    # Phase 1: Nelder-Mead optimization on rigid cluster parameters
    if time.time() < DEADLINE - 30.0:
        try:
            result = minimize(
                _objective_rigid_cluster,
                best_params.copy(),
                args=(base_positions,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.x is not None:
                cand_side = _objective_rigid_cluster(result.x, base_positions)
                if cand_side < best_side:
                    best_side = cand_side
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Extended optimization including outer center
    if time.time() < DEADLINE - 20.0:
        try:
            # Start from current best
            ext_params = np.append(best_params, [0.0, 0.0])
            
            result2 = minimize(
                _objective_with_outer_center,
                ext_params.copy(),
                args=(base_positions,),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result2.x is not None:
                cand_side = _objective_with_outer_center(result2.x, base_positions)
                if cand_side < best_side:
                    best_side = cand_side
                    best_params = result2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with smaller steps
    if time.time() < DEADLINE - 10.0:
        try:
            for restart in range(3):
                # Small random perturbation
                perturb = np.random.normal(0, 0.01, size=len(best_params))
                p = best_params + perturb
                s = _objective_with_outer_center(p, base_positions)
                if s < best_side:
                    best_side = s
                    best_params = p.copy()
                
                # Local Nelder-Mead from perturbed start
                result3 = minimize(
                    _objective_with_outer_center,
                    p.copy(),
                    args=(base_positions,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result3.x is not None:
                    cand_side = _objective_with_outer_center(result3.x, base_positions)
                    if cand_side < best_side:
                        best_side = cand_side
                        best_params = result3.x.copy()
        except Exception:
            pass
    
    # Extract final parameters
    if len(best_params) >= 5:
        cluster_angle, tx, ty, ocx, ocy = best_params
    else:
        cluster_angle, tx, ty = best_params
        ocx, ocy = 0.0, 0.0
    
    # Compute final positions
    final_positions = _rotate_and_translate(base_positions, cluster_angle, tx, ty)
    final_angles = np.full(N, cluster_angle)
    
    # Compute final outer side
    all_verts = _hex_vertices_batch(final_positions, final_angles, side=1.0)
    final_side = _min_side_vectorized(all_verts, ocx, ocy, 0.0)
    
    # Verify no overlaps
    if not _check_no_overlap(final_positions, final_angles):
        # Fallback: use simple grid
        sqrt3 = np.sqrt(3)
        fallback_positions = []
        for i in range(N):
            row = i // 3
            col = i % 3
            x = col * sqrt3 - sqrt3
            y = row * 1.5
            fallback_positions.append([x, y])
        fallback_positions = np.array(fallback_positions)
        cx = np.mean(fallback_positions[:, 0])
        cy = np.mean(fallback_positions[:, 1])
        fallback_positions -= np.array([cx, cy])
        
        final_positions = fallback_positions
        final_angles = np.zeros(N)
        all_verts = _hex_vertices_batch(final_positions, final_angles, side=1.0)
        final_side = _min_side_vectorized(all_verts, 0.0, 0.0, 0.0)
        ocx, ocy = 0.0, 0.0
    
    # Build output
    inner_list = [[float(final_positions[i][0]), float(final_positions[i][1]), float(final_angles[i])] for i in range(N)]
    
    # Final verification
    if not _check_validity(inner_list, (float(ocx), float(ocy)), float(final_side), 0.0):
        # Last resort fallback
        inner_list = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner_list, (0.0, 0.0), 40.0, 0.0
    
    return inner_list, (float(ocx), float(ocy)), float(final_side), 0.0

# EVOLVE-BLOCK-END
