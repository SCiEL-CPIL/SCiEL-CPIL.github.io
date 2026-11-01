"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon with given center, side length, and rotation."""
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def compute_required_side_fast(positions, outer_angle_deg):
    """
    Compute minimum outer hexagon side length to contain all unit hexagons at given positions.
    positions: list of (x, y) centers of unit hexagons (all oriented at 0 degrees)
    outer_angle_deg: rotation angle of the outer hexagon in degrees
    
    Returns: (side_length, cx, cy)
    """
    angle_rad = math.radians(outer_angle_deg)
    # Normals for the outer hexagon sides
    normals = []
    for k in range(6):
        theta = angle_rad + math.pi / 6 + k * math.pi / 3
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_width = 0.0
    min_projs = []
    max_projs = []
    
    for nx, ny in normals:
        # For a unit hexagon at center (px, py) with orientation 0,
        # the projection onto normal (nx, ny) ranges over:
        # (px, py) . (nx, ny) ± max_vertex_proj
        # where max_vertex_proj is the max projection of a unit hexagon's vertices onto (nx, ny)
        # For a unit hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees,
        # the max projection onto direction theta is max(cos(theta), cos(theta-60), ...)
        # Actually, for a regular hexagon of side 1, the support function in direction theta is:
        # h(theta) = max over vertices of (v . n)
        # The vertices are at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
        # h(theta) = max(cos(theta), cos(theta-pi/3), cos(theta-2pi/3), cos(theta-pi), cos(theta-4pi/3), cos(theta-5pi/3))
        # Due to symmetry, this is max(cos(theta mod pi/3 - pi/6), cos(pi/6 - theta mod pi/3))
        # Simpler: h(theta) = max(0.5 * cos(3*theta % pi - pi/2)) ... let me just compute numerically
        # Actually for a regular hexagon with circumradius 1 (side length = circumradius for regular hexagon),
        # the support function is h(theta) = max over k of cos(theta - k*pi/3)
        # But our hexagons have side length 1, so circumradius is also 1.
        
        # Compute support for each inner hexagon
        # The projection range for hexagon i is:
        # [center_i . n - h(θ_n), center_i . n + h(θ_n)]
        # where θ_n is the angle of the normal, and h is the support function
        
        # For a unit regular hexagon (side=1, circumradius=1):
        # h(φ) = max(cos(φ), cos(φ-π/3), cos(φ-2π/3), cos(φ-π), cos(φ-4π/3), cos(φ-5π/3))
        # = max(cos(φ), cos(φ+π/3), cos(φ+2π/3))  by symmetry
        
        # Let's precompute h for this normal direction
        phi = math.atan2(ny, nx)
        h_vals = [
            math.cos(phi),
            math.cos(phi - math.pi/3),
            math.cos(phi - 2*math.pi/3)
        ]
        h = max(h_vals)
        
        # Now for each inner hexagon center
        centers = np.array(positions)
        projs = centers[:, 0] * nx + centers[:, 1] * ny
        min_p = np.min(projs) - h
        max_p = np.max(projs) + h
        min_projs.append(min_p)
        max_projs.append(max_p)
        width = max_p - min_p
        if width > max_width:
            max_width = width
    
    side = max_width / SQRT3
    
    # Find center that centers the hexagon
    # Use the average of midpoints projected back
    cx = 0.0
    cy = 0.0
    for idx, (nx, ny) in enumerate(normals):
        mid_p = (min_projs[idx] + max_projs[idx]) / 2.0
        # We want c . n = mid_p
        # Solve least squares: sum of (c . n_k - mid_k)^2 = 0
        # This is an overdetermined system. Use the centroid as a good approximation.
        pass
    
    # Use centroid of positions as center
    cx = np.mean(positions, axis=0)[0]
    cy = np.mean(positions, axis=0)[1]
    
    return side, cx, cy

def compute_required_side_smooth(positions, outer_angle_deg):
    """
    Compute a smooth approximation of the required side length.
    Uses a soft-max instead of hard max for differentiability.
    """
    angle_rad = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        theta = angle_rad + math.pi / 6 + k * math.pi / 3
        normals.append((math.cos(theta), math.sin(theta)))
    
    widths = []
    for nx, ny in normals:
        phi = math.atan2(ny, nx)
        h_vals = [
            math.cos(phi),
            math.cos(phi - math.pi/3),
            math.cos(phi - 2*math.pi/3)
        ]
        h = max(h_vals)
        
        centers = np.array(positions)
        projs = centers[:, 0] * nx + centers[:, 1] * ny
        min_p = np.min(projs) - h
        max_p = np.max(projs) + h
        widths.append(max_p - min_p)
    
    # Soft max of widths
    beta = 10.0
    log_sum = np.log(np.sum(np.exp(beta * np.array(widths))))
    max_width = (log_sum - math.log(6)) / beta
    
    side = max_width / SQRT3
    cx = np.mean(positions, axis=0)[0]
    cy = np.mean(positions, axis=0)[1]
    return side, cx, cy

def check_disjoint(positions, angles):
    """Check if all unit hexagons are disjoint (no overlap)."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = positions[i]
            xj, yj = positions[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            # Minimum distance between centers of two disjoint unit hexagons
            # For two identical regular hexagons of side 1, the minimum center distance
            # for non-overlap depends on relative orientation.
            # For aligned hexagons, the minimum distance is 2 (touching at flat sides)
            # or SQRT3 * 2 / SQRT3 = 2? Let me think...
            # Two unit hexagons with flat sides parallel: min distance = 2 (center to center when touching)
            # Actually, for two unit regular hexagons (circumradius 1), the minimum center distance
            # for non-overlap is 2 when they share a flat edge.
            # But if rotated, it could be different.
            # For safety, use a conservative bound: min distance = SQRT3 (when one point touches a side?)
            # Actually, the closest approach is 2 * inradius = 2 * SQRT3/2 = SQRT3 when one vertex touches a side?
            # No, that's not right either.
            
            # Let's compute the actual minimum distance for the given orientations
            # This is expensive, so for now use a conservative check
            # For two unit hexagons, the minimum non-overlap distance is at least SQRT3
            # and at most 2 (depending on relative rotation)
            # Use 1.9 as a conservative threshold (slightly less than 2 for aligned case)
            # Actually, let's be more precise:
            # The minimum center distance for two unit hexagons to be disjoint is:
            # min over all vertex-side and vertex-vertex contacts
            # For aligned hexagons, it's 2.0 (flat side to flat side)
            # For 30 degree rotation, it could be SQRT3 ≈ 1.732 (vertex to side)
            # So the absolute minimum is SQRT3 ≈ 1.732
            
            if dist < SQRT3 - 1e-9:
                return False
    return True

def check_inside(positions, center, side, angle_deg):
    """Check if all unit hexagons are inside the outer hexagon."""
    cx, cy = center
    angle_rad = math.radians(angle_deg)
    
    # Outer hexagon vertices
    outer_verts = hex_vertices(cx, cy, side, angle_rad)
    
    # For each inner hexagon, check if all its vertices are inside the outer hexagon
    for px, py in positions:
        inner_verts = hex_vertices(px, py, 1.0, 0.0)
        for vx, vy in inner_verts:
            # Check if (vx, vy) is inside the outer hexagon
            # A point is inside a convex polygon if it's on the same side of all edges
            inside = True
            for k in range(6):
                x1, y1 = outer_verts[k]
                x2, y2 = outer_verts[(k+1) % 6]
                # Edge from (x1,y1) to (x2,y2)
                # Normal pointing inward
                ex, ey = x2 - x1, y2 - y1
                # Left normal (for CCW polygon)
                nx, ny = ey, -ex
                # Check if point is on the correct side
                edge_vec = (vx - x1, vy - y1)
                cross = nx * edge_vec[0] + ny * edge_vec[1]
                # For CCW polygon, interior is to the left of each edge
                # Need to determine orientation
                if cross < -1e-9:
                    inside = False
                    break
            if not inside:
                return False
    return True

def objective(positions, angles, outer_angle_deg):
    """
    Objective function: minimize the required outer side length.
    """
    side, cx, cy = compute_required_side_fast(positions, outer_angle_deg)
    
    # Check disjointness
    if not check_disjoint(positions, angles):
        return side + 100.0  # Large penalty
    
    # Check if inside
    if not check_inside(positions, (cx, cy), side, outer_angle_deg):
        return side + 100.0
    
    return side

def axial_to_cart(q, r):
    """Convert axial coordinates to Cartesian."""
    x = 1.5 * q
    y = SQRT3 * (r + 0.5 * q)
    return (x, y)

def construct():
    """
    Place N unit hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial configuration: center + ring 1 + 4 from ring 2
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    
    # Try different combinations of 4 from ring 2
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Generate candidate configurations
    candidates = []
    base_axial = [(0,0)] + ring1
    
    # Try all combinations of 4 from ring 2
    for i in range(len(ring2)):
        for j in range(i+1, len(ring2)):
            for k in range(j+1, len(ring2)):
                for l in range(k+1, len(ring2)):
                    combo = [ring2[i], ring2[j], ring2[k], ring2[l]]
                    shape = base_axial + combo
                    candidates.append(shape)
    
    # Also try some other compact shapes
    # Center + 2x3 block + extras
    # For now, just use the ring-based candidates
    
    for shape in candidates:
        if time.time() > deadline - 30:
            break
        
        # Convert to Cartesian
        positions = [axial_to_cart(q, r) for q, r in shape]
        angles = [0.0] * N
        
        # Try different outer hexagon orientations
        for outer_angle in range(0, 60, 5):
            if time.time() > deadline - 20:
                break
            
            # Compute required side
            side, cx, cy = compute_required_side_fast(positions, outer_angle)
            
            # Check validity
            if check_disjoint(positions, angles) and check_inside(positions, (cx, cy), side, outer_angle):
                if side < best_side:
                    best_side = side
                    best_positions = positions[:]
                    best_angles = angles[:]
                    best_center = (cx, cy)
                    best_outer_angle = float(outer_angle)
    
    # If no valid configuration found, use a fallback
    if best_positions is None:
        # Simple fallback: 11 hexagons in a more compact arrangement
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (-1, 2), (-2, 1)
        ]
        best_positions = [axial_to_cart(q, r) for q, r in axial_coords]
        best_angles = [0.0] * N
        best_outer_angle = 0.0
        best_side, best_center[0], best_center[1] = compute_required_side_fast(best_positions, best_outer_angle)
    
    # Fine-tune the outer angle
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for delta_angle in [-2, -1, 0.5, 0.2, 0.1, -0.1, -0.2, -0.5, 1, 2]:
            new_angle = best_outer_angle + delta_angle
            if new_angle < 0 or new_angle > 60:
                continue
            side, cx, cy = compute_required_side_fast(best_positions, new_angle)
            if check_disjoint(best_positions, best_angles) and check_inside(best_positions, (cx, cy), side, new_angle):
                if side < best_side - 1e-10:
                    best_side = side
                    best_center = (cx, cy)
                    best_outer_angle = new_angle
                    improved = True
                    break
        if not improved:
            break
    
    # Add tiny epsilon to side to ensure validity
    best_side += 1e-9
    
    inner = [[x, y, 0.0] for x, y in best_positions]
    return inner, best_center, best_side, best_outer_angle
