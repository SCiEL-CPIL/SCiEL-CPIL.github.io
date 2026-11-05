"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """
    Return the 6 vertices of a regular hexagon centered at (cx, cy)
    with side length 'side' and rotation angle in degrees.
    Vertices are ordered counter-clockwise.
    """
    angle_rad = math.radians(angle_deg)
    # A regular hexagon with side length s has circumradius s.
    # The vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center.
    # We rotate the whole thing by angle_rad.
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _convex_hull(points):
    """
    Compute convex hull of a set of points using a simple O(n^2) algorithm suitable for small n.
    Returns the hull vertices in counter-clockwise order.
    """
    points = np.array(points)
    if len(points) <= 3:
        return points
    
    # Find point with lowest y (and lowest x if tie)
    start_idx = np.argmin(points[:, 1])
    if len(points) > 1 and points[start_idx, 1] == points[0, 1]:
        start_idx = np.argmin(points[:, 0])
    
    hull = [start_idx]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    current = start_idx
    while True:
        candidate = (current + 1) % len(points)
        for i in range(len(points)):
            if i == current or i == candidate:
                continue
            c = cross(points[current], points[candidate], points[i])
            if c < -1e-10 or (abs(c) < 1e-10 and 
                             np.linalg.norm(points[i] - points[current]) > np.linalg.norm(points[candidate] - points[current]) + 1e-10):
                candidate = i
        if candidate == start_idx:
            break
        hull.append(candidate)
        current = candidate
        
    return points[hull]

def _min_enclosing_hexagon(hull_points):
    """
    Given a convex hull (array of points in CCW order), find the minimum side length
    of a regular hexagon that can enclose it.
    
    The minimum enclosing regular hexagon can be found by rotating the hexagon.
    For a fixed orientation, the side length L required is determined by the maximum
    projection of the hull vertices onto the 3 axes of symmetry of the hexagon.
    
    Actually, a regular hexagon is defined by 3 pairs of parallel sides.
    The distance between opposite sides is the "width" in that direction.
    For a regular hexagon with side length L, the distance between opposite sides is L * sqrt(3).
    
    So, for a given orientation of the outer hexagon, we need to find the minimum L such that
    all points are inside.
    
    The outer hexagon has 6 sides. Let's define the outer hexagon by its center and orientation.
    But the center doesn't matter for the side length if we just want the minimum size.
    Actually, the center will be the center of the bounding box in the hexagonal metric.
    
    Let's use the property that a regular hexagon is the intersection of 3 strips.
    The width of the strip in direction theta is L * sqrt(3).
    
    So, for a given orientation angle alpha of the outer hexagon, the required side length L is:
    L = max over the 3 axes of (max_projection - min_projection) / sqrt(3)
    
    Where the 3 axes are at angles alpha, alpha + 60, alpha + 120.
    Wait, the normals to the sides of a regular hexagon are at angles 0, 60, 120, 180, 240, 300.
    The distance between opposite sides (normal to side 0 and side 3) is the width in direction 0.
    
    Let's verify: A regular hexagon with side L.
    Distance from center to side is L * sqrt(3) / 2.
    So distance between opposite sides is L * sqrt(3).
    
    So if we project the hull points onto a direction n, the width is max(dot(p, n)) - min(dot(p, n)).
    We need this width <= L * sqrt(3) for all 3 pairs of opposite directions.
    So L >= width / sqrt(3) for each of the 3 directions.
    Thus L = max(width_i / sqrt(3)) for i in {0, 1, 2}.
    
    We want to minimize L over the orientation alpha.
    The function L(alpha) is periodic with period 60 degrees (pi/3).
    We can search alpha in [0, pi/3).
    """
    def compute_L(alpha):
        # 3 directions of normals
        dirs = []
        for i in range(3):
            theta = alpha + i * math.pi / 3.0
            dirs.append(np.array([math.cos(theta), math.sin(theta)]))
        
        max_width = 0.0
        for d in dirs:
            projections = hull_points @ d
            width = np.max(projections) - np.min(projections)
            if width > max_width:
                max_width = width
                
        return max_width / math.sqrt(3.0)
    
    # Search for optimal alpha
    # Use a simple grid search first, then refine
    best_L = float('inf')
    best_alpha = 0.0
    
    # Coarse search
    for alpha in np.linspace(0, math.pi / 3.0, 100, endpoint=False):
        L = compute_L(alpha)
        if L < best_L:
            best_L = L
            best_alpha = alpha
            
    # Fine search around best_alpha
    for alpha in np.linspace(best_alpha - 0.1, best_alpha + 0.1, 100):
        L = compute_L(alpha)
        if L < best_L:
            best_L = L
            best_alpha = alpha
            
    return best_L, best_alpha

# EVOLVE-BLOCK-START
def construct():
    # We have N=11 unit hexagons.
    # A good packing is a hexagonal arrangement.
    # For 11 hexagons, a 3-4-3 arrangement or similar.
    # Let's try a compact cluster.
    
    # One effective arrangement for 11 hexagons:
    # Center hexagon, surrounded by 6, and then 4 more attached to the outside.
    # Or a linear arrangement is bad.
    # A 2-3-3-2-1? No.
    
    # Let's use a known optimal or near-optimal packing for 11 hexagons.
    # The "hexagonal" packing of 11 hexagons:
    # Row 1: 2 hexagons
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Row 4: 3 hexagons
    # Total: 11.
    
    # Coordinates for unit hexagons in a triangular grid.
    # Let's place them so they are disjoint (touching is allowed).
    
    # Hexagon centers for a compact cluster:
    # (0, 0)
    # (1.5, sqrt(3)/2), (-1.5, sqrt(3)/2)
    # (1.5, -sqrt(3)/2), (-1.5, -sqrt(3)/2)
    # (3, 0), (-3, 0)
    # (1.5, 3*sqrt(3)/2), (-1.5, 3*sqrt(3)/2)
    # (1.5, -3*sqrt(3)/2), (-1.5, -3*sqrt(3)/2)
    # (0, 3*sqrt(3)/2), (0, -3*sqrt(3)/2)
    # (3, 0) ... wait, this is 19 hexagons (hexagonal number).
    
    # For 11 hexagons, let's try a 3-4-3 layout.
    # Row 1 (y = 0): 3 hexagons at x = -1.5, 0, 1.5
    # Row 2 (y = sqrt(3)): 4 hexagons at x = -2.25, -0.75, 0.75, 2.25
    # Row 3 (y = 2*sqrt(3)): 3 hexagons at x = -1.5, 0, 1.5
    
    # Let's check distances.
    # Distance between centers in same row: 1.5.
    # Side length is 1, so distance between centers of touching hexagons is 2 * sqrt(3) / 2 * ... 
    # Wait, for regular hexagons of side 1, the distance between centers of two touching hexagons is 2 * (sqrt(3)/2) * 2?
    # No. The distance from center to vertex is 1. The distance from center to side midpoint is sqrt(3)/2.
    # Two hexagons touching side-to-side: distance between centers is 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732.
    # Two hexagons touching vertex-to-vertex: distance is 2.
    
    # So for a triangular lattice of hexagons, the center-to-center distance is sqrt(3).
    # Let's adjust coordinates.
    
    # Lattice vectors:
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (1.5, -sqrt(3)/2)
    # Actually, standard hex grid:
    # x = 1.5 * col
    # y = sqrt(3) * (row + col/2)
    
    # Let's generate 11 centers.
    centers = []
    # We'll use a simple loop to create a compact cluster.
    # Start with (0,0)
    # Then add neighbors.
    
    # A better way: just define the 11 points explicitly for a known good shape.
    # Shape: A hexagon of side 2 has 7 hexagons (1 center, 6 around).
    # Add 4 more to make 11.
    
    # Let's use the 3-4-3 layout.
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 3 hexagons
    
    # Coordinates (x, y):
    # Row 0 (y=0): (-1.5, 0), (0, 0), (1.5, 0)
    # Row 1 (y=sqrt(3)): (-2.25, sqrt(3)), (-0.75, sqrt(3)), (0.75, sqrt(3)), (2.25, sqrt(3))
    # Row 2 (y=2*sqrt(3)): (-1.5, 2*sqrt(3)), (0, 2*sqrt(3)), (1.5, 2*sqrt(3))
    
    sqrt3 = math.sqrt(3)
    
    inner_points = [
        (-1.5, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (-2.25, sqrt3, 0.0),
        (-0.75, sqrt3, 0.0),
        (0.75, sqrt3, 0.0),
        (2.25, sqrt3, 0.0),
        (-1.5, 2*sqrt3, 0.0),
        (0.0, 2*sqrt3, 0.0),
        (1.5, 2*sqrt3, 0.0)
    ]
    
    # Wait, I only have 10 points. I need 11.
    # Let's add one more. Maybe a 2-4-3-2 layout?
    # Row 0: 2
    # Row 1: 4
    # Row 2: 3
    # Row 3: 2
    # Total: 11.
    
    inner_points = [
        (-0.75, 0.0, 0.0),
        (0.75, 0.0, 0.0),
        (-1.5, sqrt3, 0.0),
        (0.0, sqrt3, 0.0),
        (1.5, sqrt3, 0.0),
        (3.0, sqrt3, 0.0), # This might be too far.
        # Let's stick to a more compact shape.
    ]
    
    # Let's try a different approach. Just place them in a tight cluster.
    # The optimal packing of 11 unit circles (or hexagons) is known.
    # For hexagons, the densest packing is the hexagonal lattice.
    # A cluster of 11 hexagons from the hexagonal lattice:
    # Take a hexagon of side 2 (7 hexagons) and add 4 more.
    
    # Centers for 7 hexagons (side 2 hexagon):
    # (0,0)
    # (1.5, sqrt3/2), (-1.5, sqrt3/2)
    # (1.5, -sqrt3/2), (-1.5, -sqrt3/2)
    # (3, 0), (-3, 0)
    
    # Add 4 more:
    # (1.5, 3*sqrt3/2), (-1.5, 3*sqrt3/2)
    # (1.5, -3*sqrt3/2), (-1.5, -3*sqrt3/2)
    
    inner_points = [
        (0.0, 0.0, 0.0),
        (1.5, sqrt3/2, 0.0),
        (-1.5, sqrt3/2, 0.0),
        (1.5, -sqrt3/2, 0.0),
        (-1.5, -sqrt3/2, 0.0),
        (3.0, 0.0, 0.0),
        (-3.0, 0.0, 0.0),
        (1.5, 3*sqrt3/2, 0.0),
        (-1.5, 3*sqrt3/2, 0.0),
        (1.5, -3*sqrt3/2, 0.0),
        (-1.5, -3*sqrt3/2, 0.0)
    ]
    
    # Now, compute the convex hull of all vertices of these hexagons.
    all_vertices = []
    for x, y, angle in inner_points:
        verts = _hex_vertices(x, y, 1.0, angle)
        all_vertices.extend(verts)
        
    hull = _convex_hull(all_vertices)
    
    # Find the minimum enclosing regular hexagon.
    L, alpha = _min_enclosing_hexagon(hull)
    
    # The center of the enclosing hexagon should be the center of the hull's bounding box.
    # For a regular hexagon, the center is the average of the max and min projections.
    # Let's compute the center.
    
    # The center of the minimum enclosing hexagon is the center of the bounding box in the hexagonal metric.
    # For the optimal orientation alpha, the center is:
    # cx = (max_proj_x + min_proj_x) / 2, etc.
    
    # Let's compute the center for the optimal alpha.
    dirs = []
    for i in range(3):
        theta = alpha + i * math.pi / 3.0
        dirs.append(np.array([math.cos(theta), math.sin(theta)]))
        
    # The center is the intersection of the midlines of the 3 strips.
    # For each direction d, the midline is at (max_proj + min_proj) / 2.
    # The center (cx, cy) must satisfy dot((cx, cy), d) = midline_i for all i.
    
    # We have 3 equations and 2 unknowns. They should be consistent.
    # Let's solve for cx, cy using the first two directions.
    
    d1 = dirs[0]
    d2 = dirs[1]
    
    proj1 = hull @ d1
    mid1 = (np.max(proj1) + np.min(proj1)) / 2.0
    
    proj2 = hull @ d2
    mid2 = (np.max(proj2) + np.min(proj2)) / 2.0
    
    # Solve:
    # cx * d1[0] + cy * d1[1] = mid1
    # cx * d2[0] + cy * d2[1] = mid2
    
    A = np.array([[d1[0], d1[1]], [d2[0], d2[1]]])
    b = np.array([mid1, mid2])
    
    try:
        center = np.linalg.solve(A, b)
        cx, cy = center[0], center[1]
    except np.linalg.LinAlgError:
        cx, cy = 0.0, 0.0
        
    # The angle of the outer hexagon is alpha.
    # But the outer hexagon's orientation is defined by its vertices.
    # The normals are at alpha, alpha+60, etc.
    # The vertices are at alpha+30, alpha+90, etc.
    # The problem asks for "angle" which is the rotation of the outer hexagon.
    # Usually, this is the angle of the first vertex or the normal.
    # Let's assume it's the angle of the normal to the first side.
    
    return inner_points, (cx, cy), L, math.degrees(alpha)
# EVOLVE-BLOCK-END
