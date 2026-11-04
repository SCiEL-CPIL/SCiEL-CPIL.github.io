import math
import time
import random

N = 11
START = time.time()
DEADLINE = 145.0

# Define a set of candidate rigid clusters for 11 hexagons.
# Coordinates are in lattice units where distance between centers is 2.
# Axial coordinates (q, r) map to Cartesian (x, y) via:
# x = 2 * (q + r/2)
# y = sqrt(3) * r
# This assumes pointy-top hexagons in the lattice, but we can rotate the whole cluster.
# Actually, it's easier to just define Cartesian positions for a "flat-top" unit hexagon grid.
# Unit hexagon side 1. Distance between centers is 2.
# Flat-top hexagon vertices at angles 0, 60, ...
# Neighbors of (0,0) are at (2,0), (1, sqrt(3)), (-1, sqrt(3)), (-2,0), (-1, -sqrt(3)), (1, -sqrt(3)).

def _lattice_to_cartesian(q, r):
    # Axial to Cartesian for flat-top hexagons
    x = 2.0 * q + 1.0 * r
    y = math.sqrt(3) * r
    return x, y

# Generate candidate clusters
# Cluster 1: "Cross" or "Plus" shape (7 hexes) + 4 more
# Let's try to make a compact 3x3-ish block.
# Axial coords for a 3x3 block (actually a rhombus or hex part):
# (0,0), (1,0), (2,0)
# (0,1), (1,1), (2,1)
# (0,2), (1,2), (2,2)
# That's 9. Add (3,0) and (-1,0)? Or (3,1) and (-1,1)?
# Let's try a few specific shapes.

def _get_cluster_points(coords):
    pts = []
    for q, r in coords:
        x, y = _lattice_to_cartesian(q, r)
        pts.append((x, y))
    return pts

# Shape 1: 3x3 block (9) + 2 extensions
shape1_coords = [
    (0,0), (1,0), (2,0),
    (0,1), (1,1), (2,1),
    (0,2), (1,2), (2,2),
    (3,1), (-1,1) # Extensions to make 11
]

# Shape 2: "H" shape or similar
shape2_coords = [
    (0,0), (1,0), (2,0),
    (0,1), (1,1), (2,1),
    (0,2), (1,2), (2,2),
    (1,3), (1,-1) # Vertical extensions
]

# Shape 3: More compact, maybe a 2x5 block + 1?
shape3_coords = [
    (0,0), (1,0), (2,0), (3,0), (4,0),
    (0,1), (1,1), (2,1), (3,1), (4,1),
    (2,2) # One on top
]

# Shape 4: Center + 6 neighbors + 4 corners of the next ring
shape4_coords = [
    (0,0),
    (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1), # 7
    (2,0), (-2,0), (0,2), (0,-2) # 4 more
]

candidates = [
    _get_cluster_points(shape1_coords),
    _get_cluster_points(shape2_coords),
    _get_cluster_points(shape3_coords),
    _get_cluster_points(shape4_coords)
]

def _hex_vertices(cx, cy, s, rot_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation rot (degrees)."""
    a0 = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = a0 + math.radians(60 * k)
        pts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
    return pts

def _min_outer_side(cluster_pts, rot_deg):
    """
    Calculate the minimum side length S of an outer hexagon (centered at 0,0, rotation 0)
    that contains the cluster_pts (which are already rotated by rot_deg).
    The outer hexagon has apothem A = S * sqrt(3) / 2.
    Condition: for all points p in cluster, and for all 6 normals n_i of outer hex,
    |p . n_i| <= A.
    Normals of a flat-top hexagon (vertices at 0, 60, ...) are at 30, 90, 150, 210, 270, 330 degrees?
    No, if vertices are at 0, 60, ..., the sides are perpendicular to 30, 90, etc.
    Let's verify. Vertex at (s, 0). Side to (s/2, s*sqrt(3)/2).
    Midpoint: (3s/4, s*sqrt(3)/4). Normal vector is (sqrt(3)/2, 1/2) -> angle 30 deg.
    Yes.
    """
    # Normals of the outer hexagon (fixed orientation, rotation 0)
    normals = []
    for k in range(6):
        a = math.radians(30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    max_proj = 0.0
    # Apply rotation to cluster points
    theta = math.radians(rot_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    
    for px, py in cluster_pts:
        # Rotate point
        rx = px * c - py * s
        ry = px * s + py * c
        for nx, ny in normals:
            proj = abs(rx * nx + ry * ny)
            if proj > max_proj:
                max_proj = proj
    
    # S * sqrt(3) / 2 = max_proj  => S = 2 * max_proj / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3)

def _optimize_rotation(cluster_pts):
    """
    Find the rotation angle that minimizes the outer side length.
    Uses a simple ternary search or grid search + refinement.
    """
    best_rot = 0.0
    best_side = float('inf')
    
    # Coarse search
    for i in range(0, 60, 1):
        rot = i
        side = _min_outer_side(cluster_pts, rot)
        if side < best_side:
            best_side = side
            best_rot = rot
            
    # Fine search around best_rot
    # Try 30 degrees around best_rot with 0.1 deg steps
    for i in range(-30, 31, 1):
        rot = best_rot + i * 0.1
        side = _min_outer_side(cluster_pts, rot)
        if side < best_side:
            best_side = side
            best_rot = rot
            
    # Even finer
    for i in range(-10, 11, 1):
        rot = best_rot + i * 0.01
        side = _min_outer_side(cluster_pts, rot)
        if side < best_side:
            best_side = side
            best_rot = rot
            
    return best_rot, best_side

def construct():
    best_result = None
    best_side = float('inf')
    
    # Iterate through candidate cluster shapes
    for pts in candidates:
        rot, side = _optimize_rotation(pts)
        if side < best_side:
            best_side = side
            best_rot = rot
            
            # Construct the final inner hexagons
            # We need to return (x, y, angle_degrees) for each inner hexagon.
            # The cluster points are in the "lattice" frame.
            # We rotate the whole cluster by best_rot.
            # The outer hexagon is centered at (0,0) with side best_side and rotation 0.
            # The inner hexagons are unit side length.
            # What is the angle of each inner hexagon?
            # In the lattice frame, they are aligned with the lattice.
            # Lattice frame assumes flat-top hexagons at angle 0.
            # So if we rotate the cluster by best_rot, the inner hexagons also rotate by best_rot.
            
            inner = []
            theta = math.radians(best_rot)
            c = math.cos(theta)
            s = math.sin(theta)
            for px, py in pts:
                rx = px * c - py * s
                ry = px * s + py * c
                inner.append((rx, ry, best_rot))
                
            # Outer hexagon center (0,0), side best_side, rotation 0
            outer_c = (0.0, 0.0)
            outer_s = best_side
            outer_rot = 0.0
            
            best_result = (inner, outer_c, outer_s, outer_rot)
            
    # If no result (should not happen), fallback
    if best_result is None:
        # Fallback to a large box
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_result

# Test run
if __name__ == "__main__":
    res = construct()
    print(res)
