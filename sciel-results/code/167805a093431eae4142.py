"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hexagon_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        x = center[0] + side * np.cos(theta)
        y = center[1] + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _point_in_hexagon(px, py, center, side, angle_deg):
    """Check if point (px,py) is inside the hexagon."""
    angle = np.radians(angle_deg)
    dx = px - center[0]
    dy = py - center[1]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    verts = []
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        vx = side * np.cos(theta)
        vy = side * np.sin(theta)
        verts.append((vx, vy))
    verts = np.array(verts)
    inside = True
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        ex = x2 - x1
        ey = y2 - y1
        px_ = lx - x1
        py_ = ly - y1
        cross = ex * py_ - ey * px_
        if cross < -1e-9:
            inside = False
            break
    return inside

def _poly_overlap(v1, v2):
    """Check if two convex polygons overlap."""
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = np.array([vx * ax + vy * ay for vx, vy in v1])
        proj2 = np.array([vx * ax + vy * ay for vx, vy in v2])
        max1, min1 = proj1.max(), proj1.min()
        max2, min2 = proj2.max(), proj2.min()
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False
    return True

def _min_clearance_inner_outer(inner_verts, outer_center, outer_side, outer_angle):
    """Return minimum distance from inner polygon to outer hexagon boundary."""
    min_dist = float('inf')
    outer_verts = _hexagon_vertices(outer_center, outer_side, outer_angle)
    for vx, vy in inner_verts:
        if not _point_in_hexagon(vx, vy, outer_center, outer_side, outer_angle):
            min_d = float('inf')
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                len2 = dx*dx + dy*dy
                if len2 < 1e-12:
                    continue
                t = ((vx - x1)*dx + (vy - y1)*dy) / len2
                t = max(0, min(1, t))
                cx = x1 + t * dx
                cy = y1 + t * dy
                d = np.sqrt((vx - cx)**2 + (vy - cy)**2)
                min_d = min(min_d, d)
            return -min_d
        else:
            min_d = float('inf')
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                len2 = dx*dx + dy*dy
                if len2 < 1e-12:
                    continue
                t = ((vx - x1)*dx + (vy - y1)*dy) / len2
                t = max(0, min(1, t))
                cx = x1 + t * dx
                cy = y1 + t * dy
                d = np.sqrt((vx - cx)**2 + (vy - cy)**2)
                min_d = min(min_d, d)
            min_dist = min(min_dist, min_d)
    return min_dist

def construct():
    start_time = time.time()
    
    # Define the specific packing configuration
    # 1 center + 6 ring + 4 extra
    s3 = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (s3, 0.0),
        (s3/2, 1.5),
        (-s3/2, 1.5),
        (-s3, 0.0),
        (-s3/2, -1.5),
        (s3/2, -1.5),
        (2*s3, 0.0),
        (-2*s3, 0.0),
        (s3, 3.0),
        (-s3, 3.0)
    ]
    
    # Center the configuration
    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])
    centers = [(c[0] - cx_mean, c[1] - cy_mean) for c in centers]
    
    angles = [0.0] * N
    
    # Find the minimum enclosing hexagon
    # We can fix the outer hexagon center at (0,0) and optimize side and angle
    # Or just calculate the required side length for a few angles
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Try different angles for the outer hexagon
    for oa_deg in np.linspace(0, 60, 121):
        # Calculate required side length
        # For a given angle, the constraint is that all vertices of all inner hexagons
        # must be inside the outer hexagon.
        # This is equivalent to: max distance of any inner vertex from origin, projected onto the 6 normals?
        # No, it's simpler to just binary search on side length or calculate directly.
        
        # Direct calculation:
        # The outer hexagon is defined by 6 half-planes.
        # n_k . p <= S * cos(pi/6) ? No.
        # For a hexagon with side S, the distance from center to edge (inradius) is S * sqrt(3)/2.
        # The normal vectors are at angles oa + k*pi/3.
        # For each inner vertex, check projection on all 6 normals.
        # The required inradius r_req is max over all vertices v and normals n of (n . v).
        # Then S = r_req / (sqrt(3)/2) = 2 * r_req / sqrt(3).
        
        r_req = 0.0
        oa_rad = np.radians(oa_deg)
        for cx, cy in centers:
            # Get vertices of this inner hexagon
            for i in range(6):
                theta = np.pi * i / 3.0 # Inner hexagons are axis-aligned (angle 0)
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                # Project onto all 6 outer normals
                for k in range(6):
                    norm_angle = oa_rad + k * np.pi / 3.0
                    nx = np.cos(norm_angle)
                    ny = np.sin(norm_angle)
                    proj = vx * nx + vy * ny
                    if proj > r_req:
                        r_req = proj
        side = 2.0 * r_req / np.sqrt(3)
        if side < best_side:
            best_side = side
            best_angle = oa_deg
            
    # Verify validity
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_angle
    
    valid = True
    # Check inside
    for i in range(N):
        cx, cy = centers[i]
        inner_verts = _hexagon_vertices((cx, cy), 1.0, 0.0)
        clearance = _min_clearance_inner_outer(inner_verts, outer_center, outer_side, outer_angle)
        if clearance < -1e-6:
            valid = False
            break
            
    # Check overlap
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hexagon_vertices(centers[i], 1.0, 0.0)
                v2 = _hexagon_vertices(centers[j], 1.0, 0.0)
                if _poly_overlap(v1, v2):
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Fallback to a larger known valid config
        # Use side 4.0 and scatter
        outer_side = 4.0
        outer_angle = 0.0
        # Just ensure they fit. The above config should fit in < 4.0
        # If it failed, something is wrong with the calculation, so use a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
