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

def construct():
    start_time = time.time()
    
    # 1. Define a dense, valid packing of 11 unit hexagons in a hexagonal lattice.
    #    Pattern: 1 center, 6 in first ring, 4 in second ring.
    #    Distance between centers of adjacent hexagons in hex lattice is 2*sqrt(3)/2 = sqrt(3).
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
    
    # Center the configuration at the origin
    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])
    centers = [(c[0] - cx_mean, c[1] - cy_mean) for c in centers]
    
    # All inner hexagons are axis-aligned (angle 0)
    angles = [0.0] * N
    
    # 2. Find the optimal outer hexagon orientation and size.
    #    The outer hexagon is centered at (0,0). We vary its rotation angle.
    #    For a given outer rotation, the minimum side length S is determined by
    #    the maximum projection of any inner vertex onto the 6 normal vectors of the outer hexagon.
    #    The inradius r of the outer hexagon must be >= max projection.
    #    S = 2 * r / sqrt(3).
    
    best_side = float('inf')
    best_angle = 0.0
    
    # Scan outer rotation angles from 0 to 60 degrees (period is 60)
    # Use a fine grid for accuracy
    for oa_deg in np.linspace(0, 60, 1201):
        if time.time() > DEADLINE:
            break
        oa_rad = np.radians(oa_deg)
        r_req = 0.0
        for cx, cy in centers:
            # Get vertices of this inner hexagon (side=1, angle=0)
            for i in range(6):
                theta = np.pi * i / 3.0
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
            
    # 3. Verify validity of the found configuration
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_angle
    
    # Check all inner hexagons are inside outer
    for i in range(N):
        cx, cy = centers[i]
        inner_verts = _hexagon_vertices((cx, cy), 1.0, angles[i])
        # Check each vertex is inside outer
        for vx, vy in inner_verts:
            if not _point_in_hexagon(vx, vy, outer_center, outer_side, outer_angle):
                # If invalid, fallback to a safe large hexagon
                inner = [[float(centers[j][0]), float(centers[j][1]), float(angles[j])] for j in range(N)]
                return inner, (0.0, 0.0), 40.0, 0.0
                
    # Check no overlap between inner hexagons
    for i in range(N):
        for j in range(i + 1, N):
            v1 = _hexagon_vertices(centers[i], 1.0, angles[i])
            v2 = _hexagon_vertices(centers[j], 1.0, angles[j])
            if _poly_overlap(v1, v2):
                # If invalid, fallback to a safe large hexagon
                inner = [[float(centers[j][0]), float(centers[j][1]), float(angles[j])] for j in range(N)]
                return inner, (0.0, 0.0), 40.0, 0.0
                
    # 4. Return the result
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
