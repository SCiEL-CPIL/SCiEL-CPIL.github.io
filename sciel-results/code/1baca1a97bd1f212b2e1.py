<<<<<<< SEARCH
import numpy as np
import time
import math

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def hexagons_overlap(v1, v2):
    """
    Check if two hexagons (defined by their 6 vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap.
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal to edge
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = float('inf')
        max1 = float('-inf')
        min2 = float('inf')
        max2 = float('-inf')
        
        for p in v1:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        for p in v2:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False
    return True

def is_layout_valid(inner, eps=1e-5):
    """Check if all inner hexagons are disjoint, allowing for a small epsilon gap."""
    verts_list = []
    for (ix, iy, iangle) in inner:
        # We use angle_degrees in input, but internal check can be simplified if we just check distance
        # However, SAT is robust. Let's use SAT.
        # Note: The 'inner' list contains (x, y, angle_deg).
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        verts_list.append(verts)
    
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(verts_list[i], verts_list[j]):
                return False
    return True

def compute_required_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (flat top/bottom? No, vertex at 0 is pointy right).
    # If angle=0, vertices are at 0, 60, 120...
    # The sides are perpendicular to these.
    # The outward normals for the sides of a hexagon centered at origin with side S and rotation 0:
    # The vertices are at angles 0, 60, ...
    # The edges connect v0-v1, v1-v2, etc.
    # The normal to the edge between v0(1,0) and v1(0.5, sqrt3/2) is perpendicular to (v1-v0).
    # v1-v0 = (-0.5, sqrt3/2). Normal = (sqrt3/2, 0.5) which is angle 30.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle))
    sin_a = math.sin(math.radians(outer_angle))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to align with outer hexagon axes
            # If outer hex is rotated by alpha, we rotate points by -alpha to check against standard hex
            lx = dx * cos_a + dy * sin_a  # Wait, rotation matrix for -alpha:
            # [cos(-a) -sin(-a)] [cos(a) sin(a)]
            # [sin(-a)  cos(-a)] [-sin(a) cos(a)]
            # So lx = dx cos(a) + dy sin(a)
            # ly = -dx sin(a) + dy cos(a)
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Define 11 hexagons on a hexagonal lattice
    # Axial coordinates (q, r)
    # Center (0,0)
    # Ring 1 (6 hexagons)
    # Ring 2 partial (4 hexagons)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    # Ensure we have exactly N
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    elif len(axial_coords) < N:
        # Fallback: add more from ring 2
        # This shouldn't happen for N=11 with the above list
        pass

    # 2. Convert axial to Cartesian
    # Flat-Top lattice (vertices at 0, 60, ...):
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Add epsilon to prevent floating point overlap detection issues
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    # Verify initial layout validity
    if not is_layout_valid(initial_inner):
        # Fallback to sparse layout if something unexpected happens
        initial_inner = [[i * 2.5, 0.0, 0.0] for i in range(N)]
        centers = [(i * 2.5, 0.0) for i in range(N)]

    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon
    # Try different outer rotations
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        side = compute_required_side(initial_inner, (0.0, 0.0), outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg

    # Optimize outer center
    # Centroid of inner hexagons
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Local search on outer center
    current_cx = cx_avg
    current_cy = cy_avg
    
    # Coarse grid search around centroid
    step = 0.5
    for _ in range(2):
        improved = False
        for dx in np.linspace(-step, step, 5):
            for dy in np.linspace(-step, step, 5):
                oc = (current_cx + dx, current_cy + dy)
                side = compute_required_side(best_inner, oc, best_outer_angle)
                if side < best_side - 1e-5:
                    best_side = side
                    best_outer_center = oc
                    current_cx = oc[0]
                    current_cy = oc[1]
                    improved = True
        if not improved:
            break
        step *= 0.5

    # 4. Try rotating individual inner hexagons to see if we can pack tighter?
    # For a hexagonal lattice of identical hexagons, rotating them usually doesn't help
    # unless the lattice itself is distorted, which we can't do easily without breaking the lattice structure.
    # However, we can try a few global rotations of the entire cluster relative to the outer hex.
    # This is equivalent to changing the outer angle, which we already did.
    
    # Let's try one more thing: Perturb the positions slightly?
    # Given the tightness of the lattice packing (they touch), perturbing might cause overlaps.
    # The epsilon scale ensures they are disjoint.
    
    # Add a tiny epsilon to the side length for floating point safety in the evaluator
    final_side = best_side + 1e-9
    
    return best_inner, best_outer_center, final_side, best_outer_angle
=======
import numpy as np
import time
import math
from scipy.optimize import differential_evolution

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array."""
    angles = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def project_polygon(vertices, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    projections = vertices @ axis
    return projections.min(), projections.max()

def sat_overlap_check(v1, v2):
    """
    Check if two hexagons (numpy arrays of shape (6,2)) overlap using SAT.
    Returns True if they overlap.
    """
    # Collect all edge normals from both hexagons
    axes = []
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    # Deduplicate similar axes
    unique_axes = []
    for axis in axes:
        is_dup = False
        for ua in unique_axes:
            if abs(np.dot(axis, ua)) > 0.999:
                is_dup = True
                break
        if not is_dup:
            unique_axes.append(axis)
    
    for axis in unique_axes:
        min1, max1 = project_polygon(v1, axis)
        min2, max2 = project_polygon(v2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_all_disjoint(inner_positions, inner_angles, eps=1e-6):
    """Check if all inner hexagons are disjoint."""
    n = len(inner_positions)
    verts_list = []
    for i in range(n):
        verts = get_hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        verts_list.append(verts)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j]):
                return False
    return True

def compute_overlap_penalty(verts_list, threshold=0.0):
    """
    Compute overlap penalty between all pairs of hexagons.
    Returns a scalar penalty value (0 if no overlaps).
    Uses SAT to find minimum gap; negative gap means overlap.
    """
    n = len(verts_list)
    total_penalty = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # Collect axes
            axes = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        axes.append(normal / norm)
            
            min_gap = float('inf')
            for axis in axes:
                min1, max1 = project_polygon(v1, axis)
                min2, max2 = project_polygon(v2, axis)
                gap = min(max1 - min2, max2 - min1)
                if gap < min_gap:
                    min_gap = gap
                if gap < 0:
                    break
            
            if min_gap < 0:
                total_penalty += (-min_gap) ** 2
    
    return total_penalty

def compute_container_violation(verts_list, outer_center, outer_side, outer_angle_rad):
    """
    Compute how much inner hexagons violate the outer hexagon boundary.
    Returns 0 if all inside, positive value if violations exist.
    """
    ox, oy = outer_center
    cos_a = math.cos(outer_angle_rad)
    sin_a = math.sin(outer_angle_rad)
    
    # Standard hexagon normals (for angle=0)
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    
    max_violation = 0.0
    
    for verts in verts_list:
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to outer frame
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                # Max allowed projection is S * sqrt(3)/2
                allowed = outer_side * SQRT3 / 2.0
                violation = proj - allowed
                if violation > max_violation:
                    max_violation = violation
    
    return max_violation ** 2

def objective(params):
    """
    Objective function for differential evolution.
    params: [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, ox,oy, oside, oangle]
    Returns: oside + penalty terms
    """
    n = N
    inner_positions = np.array(params[:3*n]).reshape(n, 2)
    inner_angles = np.array(params[3*n:3*n+n])  # in radians
    
    ox = params[3*n+n]
    oy = params[3*n+n+1]
    oside = params[3*n+n+2]
    oangle = params[3*n+n+3]
    
    # Compute vertices for all inner hexagons
    verts_list = []
    for i in range(n):
        verts = get_hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        verts_list.append(verts)
    
    # Overlap penalty
    overlap_pen = compute_overlap_penalty(verts_list)
    
    # Container violation
    container_pen = compute_container_violation(verts_list, (ox, oy), oside, oangle)
    
    # Total objective: minimize side length with penalties
    return oside + 100.0 * overlap_pen + 100.0 * container_pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial layout: hexagonal close packing as starting point
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    initial_positions = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        initial_positions.append([x, y])
    
    # Compute initial bounding hexagon
    cx_avg = np.mean([p[0] for p in initial_positions])
    cy_avg = np.mean([p[1] for p in initial_positions])
    
    # Try different outer angles to find best initial side
    best_initial_side = float('inf')
    best_initial_angle = 0.0
    
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        normals = []
        for i in range(6):
            theta = math.radians(30 + i * 60)
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        max_proj = 0.0
        for ix, iy in initial_positions:
            verts = get_hex_vertices(ix, iy, 1.0, 0.0)
            for v in verts:
                dx = v[0] - cx_avg
                dy = v[1] - cy_avg
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                for nx, ny in normals:
                    proj = lx * nx + ly * ny
                    if proj > max_proj:
                        max_proj = proj
        
        side = 2.0 * max_proj / SQRT3
        if side < best_initial_side:
            best_initial_side = side
            best_initial_angle = angle_rad
    
    # Initial parameters
    init_params = []
    for i in range(N):
        init_params.extend([initial_positions[i][0], initial_positions[i][1], 0.0])
    init_params.extend([cx_avg, cy_avg, best_initial_side, best_initial_angle])
    
    # Bounds for differential evolution
    # Positions: roughly within [-4, 4] x [-4, 4]
    # Angles: [0, 2*pi]
    # Outer center: [-2, 2] x [-2, 2]
    # Outer side: [3, 10]
    # Outer angle: [0, pi] (hexagon has 6-fold symmetry, so 0 to pi/3 is enough, but use 0 to pi for safety)
    
    bounds = []
    for i in range(N):
        bounds.append((-4, 4))  # x
        bounds.append((-4, 4))  # y
        bounds.append((0, 2 * math.pi))  # angle
    bounds.append((-2, 2))  # outer cx
    bounds.append((-2, 2))  # outer cy
    bounds.append((3, 10))  # outer side
    bounds.append((0, math.pi / 3))  # outer angle (6-fold symmetry)
    
    # Run differential evolution with time limit
    best_result = None
    best_obj = float('inf')
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            x0=init_params,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=True,
            init='sobol',
            workers=1
        )
        
        if result.success or result.fun < best_obj:
            best_result = result
            best_obj = result.fun
            
    except Exception as e:
        pass
    
    # Check time and potentially do local refinement
    elapsed = time.time() - start_time
    
    if best_result is not None and best_result.fun < best_initial_side:
        final_params = best_result.x
    else:
        final_params = init_params
    
    # Extract final parameters
    inner_positions = np.array(final_params[:3*N]).reshape(N, 2)
    inner_angles = np.array(final_params[3*N:3*N+N])
    ox = final_params[3*N+N]
    oy = final_params[3*N+N+1]
    oside = final_params[3*N+N+2]
    oangle = final_params[3*N+N+3]
    
    # Verify and adjust if needed
    verts_list = []
    for i in range(N):
        verts = get_hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        verts_list.append(verts)
    
    # Check for overlaps and adjust side if necessary
    overlap_pen = compute_overlap_penalty(verts_list)
    if overlap_pen > 1e-8:
        # Scale up positions slightly to remove overlaps
        scale_factor = 1.0 + 0.01 * math.sqrt(overlap_pen)
        for i in range(N):
            inner_positions[i] = inner_positions[i] * scale_factor
        # Recompute side
        verts_list = []
        for i in range(N):
            verts = get_hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
            verts_list.append(verts)
        
        # Recompute required side
        cos_a = math.cos(oangle)
        sin_a = math.sin(oangle)
        normals = []
        for i in range(6):
            theta = math.radians(30 + i * 60)
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        max_proj = 0.0
        for verts in verts_list:
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                for nx, ny in normals:
                    proj = lx * nx + ly * ny
                    if proj > max_proj:
                        max_proj = proj
        
        oside = 2.0 * max_proj / SQRT3
    
    # Ensure all hexagons are inside the outer hexagon
    container_pen = compute_container_violation(verts_list, (ox, oy), oside, oangle)
    if container_pen > 1e-8:
        # Increase side length to contain all
        oside *= (1.0 + 0.01 * math.sqrt(container_pen))
    
    # Final safety: add tiny epsilon
    oside += 1e-9
    
    # Build inner list in required format: (x, y, angle_degrees)
    inner = []
    for i in range(N):
        angle_deg = math.degrees(inner_angles[i]) % 360.0
        inner.append([inner_positions[i][0], inner_positions[i][1], angle_deg])
    
    return inner, (ox, oy), oside, math.degrees(oangle)
>>>>>>> REPLACE
