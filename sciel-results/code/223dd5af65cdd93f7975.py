"""Packing 11 unit hexagons into minimal enclosing regular hexagon."""
import math
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        a = math.radians(angle_deg + i * 60.0)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Hexagonal grid positions using axial coordinates
    # For 11 hexagons, use a compact hexagonal cluster
    # Axial coordinates (q, r) where position = (1.5*q, sqrt(3)*(r + q/2))
    # for pointy-top, or use flat-top: (sqrt(3)*(q + r/2), 1.5*r)
    
    # Let's use flat-top hexagons for simplicity
    # Flat-top hexagon: vertices at 0, 60, 120, 180, 240, 300 degrees
    # Adjacent hexagons in flat-top grid:
    #   East: (+sqrt(3), 0)
    #   West: (-sqrt(3), 0)
    #   NE: (+sqrt(3)/2, +1.5)
    #   NW: (-sqrt(3)/2, +1.5)
    #   SE: (+sqrt(3)/2, -1.5)
    #   SW: (-sqrt(3)/2, -1.5)
    
    sqrt3 = math.sqrt(3)
    
    # Define 11 positions in a compact hexagonal arrangement
    # Center at (0,0), then 6 neighbors, then 4 more
    # Using flat-top hex grid spacing
    positions = [
        (0, 0),                    # center
        (sqrt3, 0),                # E
        (-sqrt3, 0),               # W
        (sqrt3/2, 1.5),            # NE
        (-sqrt3/2, 1.5),           # NW
        (sqrt3/2, -1.5),           # SE
        (-sqrt3/2, -1.5),          # SW
        # 4 more to make 11 - extend the ring
        (2*sqrt3, 0),              # E2
        (sqrt3, 3.0),              # NE2
        (-sqrt3, 3.0),             # NW2
        (-2*sqrt3, 0),             # W2
    ]
    
    # All inner hexagons are unit (side=1), angle=0 (flat-top)
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Now find the minimal enclosing regular hexagon
    # The outer hexagon can be rotated; we need to find center, side, and rotation
    # that contains all 11 hexagons
    
    # Strategy: The bounding region of the hexagons is the union of all hexagon vertices
    # We need the smallest regular hexagon (any rotation) that contains all these points
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for x, y, _ in inner:
        for i in range(6):
            a = math.radians(i * 60.0)  # flat-top: 0, 60, 120, 180, 240, 300
            px = x + math.cos(a)
            py = y + math.sin(a)
            all_points.append((px, py))
    
    # Find centroid of all points as initial center estimate
    cx = sum(p[0] for p in all_points) / len(all_points)
    cy = sum(p[1] for p in all_points) / len(all_points)
    
    # For a given rotation angle of the outer hexagon, find the minimal side length
    # that contains all points. The outer hexagon with center (cx,cy), side S, rotation theta
    # has vertices at distance S from center at angles theta + i*60.
    # A point is inside if its projection onto each of the 6 normal directions is <= S*sqrt(3)/2
    # (the apothem of the outer hexagon)
    
    def min_side_for_rotation(theta_deg, cx, cy):
        """Find minimal side length of outer hexagon with given rotation containing all points."""
        max_apothem = 0.0
        for px, py in all_points:
            dx = px - cx
            dy = py - cy
            # Check projection onto each of 6 normal directions
            for i in range(6):
                # Normal direction for face i of hexagon rotated by theta
                # The face normals are at angles theta + 30 + i*60 (perpendicular to edges)
                # Actually for a regular hexagon, the apothem direction for face i is at angle theta + i*60
                # Let me reconsider: vertices at theta + i*60, so edge midpoints (apothem directions) at theta + 30 + i*60
                normal_angle = math.radians(theta_deg + 30 + i * 60)
                proj = dx * math.cos(normal_angle) + dy * math.sin(normal_angle)
                if proj > max_apothem:
                    max_apothem = proj
        # Apothem = S * sqrt(3)/2, so S = max_apothem * 2 / sqrt(3)
        return max_apothem * 2.0 / sqrt3
    
    # Search over rotation angles
    best_side = float('inf')
    best_theta = 0.0
    best_cx, best_cy = cx, cy
    
    # Coarse search over rotation
    for theta_deg in range(0, 60, 1):
        if time.time() > deadline:
            break
        s = min_side_for_rotation(theta_deg, cx, cy)
        if s < best_side:
            best_side = s
            best_theta = theta_deg
    
    # Fine search around best
    if time.time() < deadline - 5:
        for theta_deg in [best_theta + d/10.0 for d in range(-10, 11)]:
            if time.time() > deadline:
                break
            s = min_side_for_rotation(theta_deg, cx, cy)
            if s < best_side:
                best_side = s
                best_theta = theta_deg
    
    # Try a few different center positions
    if time.time() < deadline - 10:
        # Try shifting center slightly
        for dx in [-0.5, -0.25, 0, 0.25, 0.5]:
            for dy in [-0.5, -0.25, 0, 0.25, 0.5]:
                if time.time() > deadline:
                    break
                tcx, tcy = cx + dx, cy + dy
                for theta_deg in [best_theta, best_theta + 5, best_theta - 5]:
                    if time.time() > deadline:
                        break
                    s = min_side_for_rotation(theta_deg, tcx, tcy)
                    if s < best_side:
                        best_side = s
                        best_theta = theta_deg
                        best_cx, best_cy = tcx, tcy
    
    return inner, (best_cx, best_cy), best_side, best_theta

# EVOLVE-BLOCK-END
