import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3)

def get_hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a regular unit hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.linspace(0, 2 * np.pi, 7)[:-1]
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def project_polygon(vertices, axis):
    """Project polygon vertices onto an axis."""
    projections = vertices @ axis
    return np.min(projections), np.max(projections)

def hexagons_overlap(v1, v2):
    """
    Check if two hexagons (defined by their 6 vertices) overlap using SAT.
    v1, v2: (6, 2) numpy arrays.
    Returns True if they overlap.
    """
    axes = []
    # Get axes from edges of v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        # Normal to edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            axes.append(normal / norm)
    
    # Get axes from edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            axes.append(normal / norm)

    for axis in axes:
        min1, max1 = project_polygon(v1, axis)
        min2, max2 = project_polygon(v2, axis)
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False
    return True

def is_layout_valid(inner):
    """Check if all inner hexagons are disjoint."""
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, iangle)
        verts_list.append(verts)
    
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(verts_list[i], verts_list[j]):
                return False
    return True

def is_point_in_hexagon(pt, cx, cy, side, angle_deg):
    """Check if a point is inside a regular hexagon."""
    dx = pt[0] - cx
    dy = pt[1] - cy
    angle_rad = -np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    normals = np.deg2rad([30, 90, 150, 210, 270, 330])
    projections = lx * np.cos(normals) + ly * np.sin(normals)
    return np.all(projections <= side * SQRT3 / 2)

def compute_required_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = np.deg2rad([30, 90, 150, 210, 270, 330])
    cos_normals = np.cos(normals)
    sin_normals = np.sin(normals)
    
    max_projection = 0.0
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, iangle)
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            angle_rad = -np.deg2rad(outer_angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            projections = lx * cos_normals + ly * sin_normals
            max_projection = max(max_projection, np.max(projections))
    
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initial layout: Linear arrangement with Angle 30, Spacing SQRT3
    # This ensures they touch edge-to-edge and are disjoint.
    inner = [[i * SQRT3, 0.0, 30.0] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Verify initial layout
    if not is_layout_valid(inner):
        # Fallback to very sparse layout if something goes wrong
        inner = [[i * 2.5, 0.0, 0.0] for i in range(N)]

    best_side = compute_required_side(inner, outer_center, outer_angle)
    best_inner = inner
    best_outer_center = outer_center
    best_outer_angle = outer_angle

    # Try different outer hexagon rotations
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        side = compute_required_side(best_inner, best_outer_center, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg

    # Try a 2D arrangement
    # Use Angle 30 for inner hexagons to allow tighter packing (spacing SQRT3)
    # We will try to pack them in a hexagonal grid pattern.
    # For Angle 30 (flat top/bottom relative to vertical? No, flat left/right?
    # Angle 0: Vertex Right.
    # Angle 30: Flat Right.
    # If we use Angle 30, the "width" is SQRT3, "height" is 2.
    # Horizontal spacing: SQRT3.
    # Vertical spacing: 1.5?
    # Let's just generate candidate positions and check validity.

    # Generate a grid of candidate positions
    # Centered around 0,0
    # Spacing SQRT3 in X, 1.5 in Y (for hex grid of flat-top hexes)
    # Actually, for flat-right hexes (Angle 30):
    # X neighbors: SQRT3
    # Y neighbors: 1.5 (offset by SQRT3/2 in X)
    
    candidates = []
    # Generate a larger grid than needed
    max_radius = 5
    for x in np.linspace(-max_radius * SQRT3, max_radius * SQRT3, 40):
        for y in np.linspace(-max_radius * 1.5, max_radius * 1.5, 40):
            candidates.append((x, y))
    
    # Shuffle candidates? Or just pick the N closest to center?
    # Better: Pick N positions from a valid hexagonal lattice subset.
    # Let's construct a valid hexagonal lattice of Angle 30 hexagons.
    # Lattice vectors:
    # u = (SQRT3, 0)
    # v = (SQRT3/2, 1.5)
    
    lattice_points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * SQRT3 + j * (SQRT3 / 2)
            y = j * 1.5
            lattice_points.append((x, y))
            
    # Sort by distance to origin to pick the densest central cluster
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Try the first N points
    if len(lattice_points) >= N:
        initial_2d = [[x, y, 30.0] for x, y in lattice_points[:N]]
        if is_layout_valid(initial_2d):
            for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
                side = compute_required_side(initial_2d, (0.0, 0.0), outer_angle_deg)
                if side < best_side:
                    best_side = side
                    best_inner = initial_2d
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = outer_angle_deg

    # Optimize outer center
    if time.time() < deadline - 10:
        # Centroid of best_inner
        all_verts = []
        for (ix, iy, iangle) in best_inner:
            verts = get_hex_vertices(ix, iy, iangle)
            all_verts.extend(verts)
        all_verts = np.array(all_verts)
        centroid = np.mean(all_verts, axis=0)
        
        best_side2 = compute_required_side(best_inner, tuple(centroid), best_outer_angle)
        if best_side2 < best_side:
            best_side = best_side2
            best_outer_center = tuple(centroid)
            
        # Local search on outer center
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            improved = False
            for dx in np.linspace(-0.5, 0.5, 5):
                for dy in np.linspace(-0.5, 0.5, 5):
                    oc = (best_outer_center[0] + dx, best_outer_center[1] + dy)
                    side = compute_required_side(best_inner, oc, best_outer_angle)
                    if side < best_side - 1e-5:
                        best_side = side
                        best_outer_center = oc
                        improved = True
            if not improved:
                break

    return best_inner, best_outer_center, best_side, best_outer_angle
