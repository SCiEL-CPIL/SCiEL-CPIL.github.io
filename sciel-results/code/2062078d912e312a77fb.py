import numpy as np
import time

N = 11
DEADLINE = 140.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_convex_polygon(pt, poly, tol=1e-9):
    """Check if point is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > tol:
            if sign < 0:
                return False
            sign = 1
        elif cross < -tol:
            if sign > 0:
                return False
            sign = -1
    return True

def _verify_packing(inner, outer_center, side, outer_angle):
    """
    Verify that all inner hexagons are inside the outer hexagon and don't overlap.
    Uses tolerance to allow touching (interiors disjoint).
    Returns True if valid, False otherwise.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_angle, side)
    
    inner_verts_list = []
    for x, y, a in inner:
        verts = _hex_vertices(x, y, a, 1.0)
        inner_verts_list.append(verts)
        # Check containment: all vertices of inner hex must be inside outer hex
        for v in verts:
            if not _point_in_convex_polygon(v, outer_verts, tol=1e-8):
                return False
    
    # Check pairwise non-overlap: no vertex of one hex strictly inside another
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            v1 = inner_verts_list[i]
            v2 = inner_verts_list[j]
            for v in v1:
                if _point_in_convex_polygon(v, v2, tol=1e-8):
                    return False
            for v in v2:
                if _point_in_convex_polygon(v, v1, tol=1e-8):
                    return False
    return True

def construct():
    start_time = time.time()
    
    # Deterministic "flower" packing: 1 center + 6 ring1 + 4 ring2
    # All inner hexagons rotated 30 degrees (pointy-top)
    # Use minimal epsilon for numerical safety
    eps = 1e-10
    
    centers = []
    
    # Ring 0: Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    # Ring 2: 4 hexagons at distance 3, angles 30, 90, 150, 270
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    
    inner_angle = 30.0
    inner = []
    for x, y in centers:
        inner.append([x, y, inner_angle])
    
    # Calculate required outer side length analytically
    # For pointy-top hexagons (angle 30°):
    #   x-extent per hex: sqrt(3)/2 (vertices at 30°, 330°)
    #   y-extent per hex: 1.0 (vertex at 90°)
    max_x = 0.0
    max_y = 0.0
    for x, y in centers:
        max_x = max(max_x, abs(x) + np.sqrt(3) / 2)
        max_y = max(max_y, abs(y) + 1.0)
    
    # Outer hexagon pointy-top (angle 30°):
    #   width (x) = S * sqrt(3)
    #   height (y) = 2 * S
    s_from_x = 2 * max_x / np.sqrt(3)
    s_from_y = max_y
    side = max(s_from_x, s_from_y) + 1e-9
    
    outer_center = (0.0, 0.0)
    outer_angle = 30.0
    
    # Quick verification with time guard
    if time.time() - start_time < DEADLINE - 5:
        if not _verify_packing(inner, outer_center, side, outer_angle):
            # If verification fails, add small margin and retry
            side += 1e-6
            if not _verify_packing(inner, outer_center, side, outer_angle):
                # Last resort: slightly larger
                side += 1e-4
    
    return inner, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
