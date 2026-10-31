import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Hexagonal grid arrangement for 11 unit hexagons
    # All same orientation (0 degrees)
    # Pattern: 4-3-4 rows
    sqrt3 = math.sqrt(3)
    
    # Centers in hexagonal grid
    centers = []
    # Row 0: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * sqrt3
        y = 0.0
        centers.append((x, y, 0.0))
    # Row 1: 3 hexagons (offset)
    for i in range(3):
        x = (i - 1.0) * sqrt3
        y = 1.5
        centers.append((x, y, 0.0))
    # Row 2: 4 hexagons
    for i in range(4):
        x = (i - 1.5) * sqrt3
        y = 3.0
        centers.append((x, y, 0.0))
    
    inner = centers
    
    # Compute minimum enclosing regular hexagon
    # For a regular hexagon with side S, center (cx,cy), rotation theta:
    # A point is inside if max over 6 sides of dot(point-center, normal_i) <= S*sqrt(3)/2
    
    def hexagon_contains(cx, cy, S, theta, px, py):
        """Check if point (px,py) is inside regular hexagon with side S, center (cx,cy), rotation theta."""
        dx = px - cx
        dy = py - cy
        inradius = S * sqrt3 / 2
        for k in range(6):
            angle = theta + k * math.pi / 3
            nx = math.cos(angle)
            ny = math.sin(angle)
            proj = dx * nx + dy * ny
            if proj > inradius + 1e-9:
                return False
        return True
    
    def min_side_length(cx, cy, theta, points):
        """Find minimum side length of hexagon with center (cx,cy), rotation theta that contains all points."""
        max_proj = 0
        for k in range(6):
            angle = theta + k * math.pi / 3
            nx = math.cos(angle)
            ny = math.sin(angle)
            for px, py in points:
                proj = (px - cx) * nx + (py - cy) * ny
                if proj > max_proj:
                    max_proj = proj
        return 2 * max_proj / sqrt3
    
    # The outer hexagon must contain all inner hexagons, not just centers
    # Each inner hexagon (side 1, orientation 0) has vertices at distance 1 from center
    # So we need to check all vertices of each inner hexagon
    
    def inner_hex_vertices(cx, cy, angle_deg):
        """Get 6 vertices of unit hexagon with center (cx,cy), rotation angle_deg."""
        verts = []
        for k in range(6):
            angle = math.radians(angle_deg) + k * math.pi / 3
            verts.append((cx + math.cos(angle), cy + math.sin(angle)))
        return verts
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner:
        verts = inner_hex_vertices(cx, cy, ang)
        all_verts.extend(verts)
    
    # Try different rotations and centers to find minimum enclosing hexagon
    best_S = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_theta = 0.0
    
    # The center of the enclosing hexagon should be near the centroid of the points
    cx_centroid = sum(p[0] for p in all_verts) / len(all_verts)
    cy_centroid = sum(p[1] for p in all_verts) / len(all_verts)
    
    # Try various rotations
    for theta_deg in range(0, 60, 3):
        if time.time() > deadline:
            break
        theta = math.radians(theta_deg)
        S = min_side_length(cx_centroid, cy_centroid, theta, all_verts)
        if S < best_S:
            best_S = S
            best_cx = cx_centroid
            best_cy = cy_centroid
            best_theta = theta
    
    # Also try offsetting the center slightly
    for dcx in [-0.5, 0, 0.5]:
        for dcy in [-0.5, 0, 0.5]:
            if time.time() > deadline:
                break
            for theta_deg in range(0, 60, 5):
                theta = math.radians(theta_deg)
                S = min_side_length(cx_centroid + dcx, cy_centroid + dcy, theta, all_verts)
                if S < best_S:
                    best_S = S
                    best_cx = cx_centroid + dcx
                    best_cy = cy_centroid + dcy
                    best_theta = theta
    
    # Convert theta back to degrees for output
    best_theta_deg = math.degrees(best_theta)
    
    return inner, (best_cx, best_cy), best_S, best_theta_deg
# EVOLVE-BLOCK-END
