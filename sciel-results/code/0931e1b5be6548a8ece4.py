"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside or on boundary of hexagon."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = -np.radians(angle_deg)
    # Rotate point into hexagon frame
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s centered at origin, aligned with flat top:
    # The hexagon can be described by |x| <= s and |x|/s + |y|/(s*sqrt(3)/2) <= 1... 
    # Actually, let's use the 6 half-plane constraints.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center, distance s from center? 
    # No! For a regular hexagon with side length s, the circumradius (center to vertex) = s.
    # The vertices are at distance s from center.
    # So in local frame, vertices are at (s*cos(i*pi/3), s*sin(i*pi/3)) for i=0..5.
    
    # The hexagon is the intersection of 6 half-planes.
    # For each edge, the inward normal points toward center.
    # Edge between vertex i and vertex i+1.
    # The distance from center to each edge (apothem) = s * sqrt(3)/2.
    
    # Let's check using the 6 half-plane inequalities.
    # In the local frame, the hexagon with vertices at angle i*pi/3 has edges.
    # The normal to the edge between vertices at angle (i-1)*pi/3 and i*pi/3... 
    
    # Simpler: use the fact that for a hexagon with circumradius R=s,
    # a point (x,y) is inside if for all i=0..5:
    # x*cos(i*pi/3) + y*sin(i*pi/3) <= s * sqrt(3)/2
    # Wait, that's not quite right either.
    
    # Let me think again. Vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5.
    # The edge between vertex k and vertex k+1 (mod 6).
    # The midpoint of this edge is at distance apothem = s*sqrt(3)/2 from center, 
    # in the direction of angle (k+0.5)*pi/3.
    # The inward normal to this edge points from the edge midpoint toward center, 
    # which is in the direction of angle (k+0.5)*pi/3 + pi = (k+2.5)*pi/3.
    # Actually, the outward normal is in the direction of the midpoint from center: angle (k+0.5)*pi/3.
    # So the half-plane constraint is:
    # x*cos((k+0.5)*pi/3) + y*sin((k+0.5)*pi/3) <= s*sqrt(3)/2
    
    apothem = s * np.sqrt(3) / 2 if (s := side) else 0
    for k in range(6):
        a = (k + 0.5) * np.pi / 3
        val = lx * np.cos(a) + ly * np.sin(a)
        if val > side * np.sqrt(3) / 2 + 1e-10:
            return False
    return True

def _all_points_in_hexagon(points, cx, cy, side, angle_deg):
    """Check if all points are inside the hexagon."""
    for px, py in points:
        if not _point_in_hexagon(px, py, cx, cy, side, angle_deg):
            return False
    return True

def _hexagon_extreme_points(cx, cy, side, angle_deg):
    """Get the 6 vertices of a hexagon (these are the extreme points)."""
    return _hexagon_vertices(cx, cy, side, angle_deg)

def _min_containing_hexagon_side(inner_hexes, outer_cx, outer_cy, outer_angle_deg):
    """
    Find minimum side length of regular hexagon centered at (outer_cx, outer_cy) 
    with rotation outer_angle_deg that contains all inner hexagons.
    Each inner hexagon is defined by (cx, cy, angle_deg) with side=1.
    """
    # For each inner hexagon, get its 6 vertices
    all_points = []
    for cx, cy, ang in inner_hexes:
        verts = _hexagon_vertices(cx, cy, 1.0, ang)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    # Transform all points to outer hexagon's local frame
    dx = all_points[:, 0] - outer_cx
    dy = all_points[:, 1] - outer_cy
    angle = -np.radians(outer_angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon with side s has constraint:
    # lx*cos((k+0.5)*pi/3) + ly*sin((k+0.5)*pi/3) <= s*sqrt(3)/2 for k=0..5
    # So s >= (lx*cos(a_k) + ly*sin(a_k)) / (sqrt(3)/2) for all k and all points
    # i.e., s >= max over all points and k of (lx*cos(a_k) + ly*sin(a_k)) * 2/sqrt(3)
    
    max_val = 0.0
    for k in range(6):
        a = (k + 0.5) * np.pi / 3
        vals = lx * np.cos(a) + ly * np.sin(a)
        max_val = max(max_val, np.max(vals))
    
    return max_val * 2.0 / np.sqrt(3)

def construct():
    # Define 11 hexagons in a compact hexagonal arrangement
    # Center at origin, 6 in first ring, 4 in second ring
    d = np.sqrt(3)  # distance between adjacent hexagon centers
    
    inner = [[0.0, 0.0, 0.0]]
    
    # First ring: 6 hexagons
    for i in range(6):
        a = i * np.pi / 3
        x = d * np.cos(a)
        y = d * np.sin(a)
        inner.append([x, y, 0.0])
    
    # Second ring: pick 4 positions to keep it compact
    # Second ring positions are at distance 2*d in the 6 directions, 
    # or at distance d*sqrt(3) in the 12 intermediate directions... 
    # Actually for hexagonal lattice, second ring has 12 positions.
    # Let's place 4 in the second ring at cardinal-ish positions to minimize bounding box.
    
    # Positions in second ring (distance 2d from center in the 6 main directions,
    # plus 6 more at distance d*sqrt(3) in intermediate directions)
    # For compactness, place 4 at the "corners" that are less protruding.
    
    # Let's try: place 4 at angles 0, 120, 240, and one more... 
    # Actually, let me just try a few configurations and pick the best.
    
    # Configuration: add 4 hexagons at second-ring positions
    # Second ring: 12 positions. 6 at distance 2d, angles 0,60,120,180,240,300
    # 6 at distance d*sqrt(3), angles 30,90,150,210,270,330
    
    # Try placing 4 at the 6 "far" positions (distance 2d) - pick 4 of them
    # To minimize bounding hexagon, pick 4 that are somewhat symmetric.
    
    # Let's try: add at angles 0, 120, 240 (3 hexagons) + 1 more
    # Or just try all combinations of 4 from 12 positions? That's C(12,4)=495, feasible.
    
    # Generate all 12 second-ring positions
    second_ring = []
    for i in range(6):
        a = i * np.pi / 3
        x = 2 * d * np.cos(a)
        y = 2 * d * np.sin(a)
        second_ring.append((x, y))
    for i in range(6):
        a = (i + 0.5) * np.pi / 3
        x = d * np.sqrt(3) * np.cos(a)
        y = d * np.sqrt(3) * np.sin(a)
        second_ring.append((x, y))
    
    # Try all combinations of 4 from 12
    from itertools import combinations
    best_side = float('inf')
    best_config = None
    best_outer_angle = 0.0
    
    # Also optimize outer hexagon rotation
    for combo in combinations(range(12), 4):
        inner_test = inner.copy()
        for idx in combo:
            x, y = second_ring[idx]
            inner_test.append([x, y, 0.0])
        
        # Try several outer rotations
        for outer_ang in [0, 15, 30, 45, 60]:
            side = _min_containing_hexagon_side(inner_test, 0.0, 0.0, outer_ang)
            if side < best_side:
                best_side = side
                best_config = inner_test
                best_outer_angle = outer_ang
    
    # Further optimize outer rotation with fine search
    if best_config is not None:
        best_side_refined = best_side
        best_outer_angle_refined = best_outer_angle
        for outer_ang in np.linspace(0, 60, 61):
            side = _min_containing_hexagon_side(best_config, 0.0, 0.0, outer_ang)
            if side < best_side_refined:
                best_side_refined = side
                best_outer_angle_refined = outer_ang
        
        # Even finer
        if best_outer_angle_refined > 0 and best_outer_angle_refined < 60:
            for outer_ang in np.linspace(best_outer_angle_refined - 2, best_outer_angle_refined + 2, 41):
                side = _min_containing_hexagon_side(best_config, 0.0, 0.0, outer_ang)
                if side < best_side_refined:
                    best_side_refined = side
                    best_outer_angle_refined = outer_ang
        
        best_side = best_side_refined
        best_outer_angle = best_outer_angle_refined
    
    return best_config, (0.0, 0.0), best_side, best_outer_angle
# EVOLVE-BLOCK-END

