"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree
import itertools

N = 11

def get_hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon with center (cx, cy), rotation angle_deg, side length."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(vertices)

def polygon_distance(v1, v2):
    """Compute minimum distance between two convex polygons (hexagons)."""
    # For convex polygons, min distance is min of:
    # 1. Distances from vertices of one to edges of other
    # 2. Distances from vertices of other to edges of one
    # 3. Zero if they intersect
    
    # Check if polygons intersect using a simpler approach
    # For hexagons, we can check if any edge pair intersects
    
    def seg_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    def point_segment_dist(px, py, x1, y1, x2, y2):
        """Distance from point to segment."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        qx = x1 + t * dx
        qy = y1 + t * dy
        return np.sqrt((px - qx)**2 + (py - qy)**2)
    
    # Check intersection
    n1 = len(v1)
    n2 = len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            p3 = v2[j]
            p4 = v2[(j + 1) % n2]
            if seg_intersect(p1, p2, p3, p4):
                return 0.0
    
    # Compute minimum distance
    min_dist = float('inf')
    for i in range(n1):
        for j in range(n2):
            d = point_segment_dist(v1[i, 0], v1[i, 1], v2[j, 0], v2[j, 1], v2[(j + 1) % n2, 0], v2[(j + 1) % n2, 1])
            min_dist = min(min_dist, d)
            d = point_segment_dist(v2[j, 0], v2[j, 1], v1[i, 0], v1[i, 1], v1[(i + 1) % n1, 0], v1[(i + 1) % n1, 1])
            min_dist = min(min_dist, d)
    
    return min_dist

def are_disjoint(pos1, ang1, pos2, ang2, eps=1e-9):
    """Check if two unit hexagons are disjoint (allowing touching within eps)."""
    v1 = get_hex_vertices(pos1[0], pos1[1], ang1)
    v2 = get_hex_vertices(pos2[0], pos2[1], ang2)
    dist = polygon_distance(v1, v2)
    return dist >= -eps  # Allow tiny overlap for numerical tolerance

def check_all_disjoint(positions, angles, eps=1e-9):
    """Check if all hexagons are pairwise disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if not are_disjoint(positions[i], angles[i], positions[j], angles[j], eps):
                return False
    return True

def compute_required_side(positions, angles, outer_center, outer_angle):
    """Compute minimum outer hexagon side length needed to contain all inner hexagons."""
    outer_center = np.array(outer_center)
    
    # For each inner hexagon, find the maximum distance from outer center to any vertex
    # The outer hexagon must contain all these points
    # A regular hexagon with side s and rotation theta contains a point (x, y) if
    # the point is within the hexagon boundary
    
    # Actually, let's think differently. The outer hexagon with center C, side s, rotation theta
    # has vertices at C + s * (cos(theta + k*pi/3), sin(theta + k*pi/3)) for k=0..5
    
    # A point P is inside this hexagon if it can be expressed as a convex combination of vertices,
    # or equivalently, if it satisfies all 6 half-plane constraints.
    
    # For a regular hexagon with center at origin and rotation theta, the edges are at distance 
    # s * cos(pi/6) = s * sqrt(3)/2 from center. The normal to edge k points at angle theta + k*pi/3.
    # So constraint k: dot(P - C, n_k) <= s * sqrt(3)/2 where n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3))
    
    # Wait, let me reconsider. For a regular hexagon with side length s centered at origin:
    # The apothem (distance from center to edge midpoint) is s * sqrt(3)/2
    # The vertices are at distance s from center
    
    # The 6 edge normals are at angles theta + k*pi/3 for k=0..5
    # Constraint: for each k, dot(P, n_k) <= apothem = s * sqrt(3)/2
    
    # So for all inner hexagon vertices P, we need:
    # max over all P and k of dot(P - C, n_k) <= s * sqrt(3)/2
    
    # Thus s >= (2/sqrt(3)) * max over all P and k of dot(P - C, n_k)
    
    theta = np.radians(outer_angle)
    max_dot = 0.0
    
    for i in range(len(positions)):
        v = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
        for k in range(6):
            nk_angle = theta + k * np.pi / 3
            n_k = np.array([np.cos(nk_angle), np.sin(nk_angle)])
            for vertex in v:
                d = np.dot(vertex - outer_center, n_k)
                if d > max_dot:
                    max_dot = d
    
    return (2.0 / np.sqrt(3.0)) * max_dot

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10 second safety margin
    
    # Generate hexagonal lattice positions for N=11 hexagons
    # Center + 6 in first ring + 4 more in second ring (or arrange optimally)
    
    # For 11 hexagons, a good arrangement is:
    # 1 center, 6 in first ring (distance = 2 for unit hexagons to be disjoint)
    # 4 in second ring at optimal positions
    
    # Actually, for unit regular hexagons to be disjoint on a hexagonal lattice,
    # adjacent centers need to be at least 2*sqrt(3) apart? No...
    # For two unit hexagons to be disjoint (non-overlapping), their centers need to be far enough.
    # The minimum center-to-center distance for two unit hexagons to just touch is 2*sqrt(3) if aligned,
    # but it depends on relative orientation.
    
    # For hexagons with same orientation, the minimum distance between centers for them to be
    # disjoint is 2 * sqrt(3) (when they share an edge).
    
    # Let's use a hexagonal lattice with spacing 2*sqrt(3) + small gap
    
    s = np.sqrt(3.0)
    gap = 0.01
    spacing = 2 * s + gap
    
    # Generate positions in hexagonal coordinate system
    # For N=11: center (0,0), 6 at distance spacing, 4 more at 2*spacing in some directions
    
    positions = []
    angles = []
    
    # Center hexagon
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # First ring: 6 hexagons
    for k in range(6):
        theta = k * np.pi / 3
        x = spacing * np.cos(theta)
        y = spacing * np.sin(theta)
        positions.append((x, y))
        angles.append(0.0)
    
    # Second ring: need 4 more. Place them at 2*spacing in 4 of the 6 directions
    # Or place them between first ring hexagons
    # Actually, for a hexagonal packing, the second ring has 12 positions, but we only need 4
    # Let's place them at 2*spacing in directions that maximize symmetry
    
    # Better: use axial coordinates. For a hexagonal lattice, positions are:
    # (q, r) -> (x, y) = (spacing * (q + r/2), spacing * r * sqrt(3)/2)
    
    # Let's generate all positions within a hexagonal region of radius 2 in axial coords
    # and pick the best 11
    
    def axial_to_xy(q, r, sp):
        x = sp * (q + r / 2.0)
        y = sp * r * s / 2.0
        return (x, y)
    
    # Generate candidate positions in axial coordinates within distance 2
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check if within hexagonal distance 2
            dq = abs(q)
            dr = abs(r)
            ds = abs(q + r)
            if max(dq, dr, ds) <= 2:
                candidates.append((q, r))
    
    # Sort by distance from center to get compact arrangement
    def hex_dist(q, r):
        return (abs(q) + abs(r) + abs(q + r)) / 2.0
    
    candidates.sort(key=lambda c: (hex_dist(c[0], c[1]), c[0], c[1]))
    
    # Take first 11 candidates
    best_positions = []
    for q, r in candidates[:11]:
        x, y = axial_to_xy(q, r, spacing)
        best_positions.append((x, y))
    
    # Now optimize: try different configurations
    best_side = float('inf')
    best_outer_angle = 0.0
    best_config = None
    
    # Try different outer hexagon rotations
    for outer_angle in range(0, 60, 5):
        pos_list = [p for p in best_positions]
        ang_list = [0.0] * len(pos_list)
        side = compute_required_side(pos_list, ang_list, (0.0, 0.0), outer_angle)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle
            best_config = (pos_list, ang_list)
    
    # Fine-tune outer angle
    if time.time() < deadline - 30:
        for outer_angle in np.arange(best_outer_angle - 5, best_outer_angle + 5, 0.5):
            pos_list = [p for p in best_positions]
            ang_list = [0.0] * len(pos_list)
            side = compute_required_side(pos_list, ang_list, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_outer_angle = outer_angle
                best_config = (pos_list, ang_list)
    
    # Try rotating inner hexagons to reduce required outer size
    if time.time() < deadline - 30:
        best_inner_rot = 0.0
        for inner_rot in range(0, 60, 15):
            pos_list = [p for p in best_positions]
            ang_list = [inner_rot] * len(pos_list)
            for outer_angle in [best_outer_angle, best_outer_angle + 30]:
                side = compute_required_side(pos_list, ang_list, (0.0, 0.0), outer_angle)
                if side < best_side:
                    best_side = side
                    best_outer_angle = outer_angle
                    best_inner_rot = inner_rot
                    best_config = (pos_list, [inner_rot] * len(pos_list))
    
    # Try adjusting spacing (making it slightly smaller to see if still valid)
    if time.time() < deadline - 20:
        for scale in [1.0, 0.99, 0.98, 0.97]:
            sp = spacing * scale
            test_positions = []
            for q, r in candidates[:11]:
                x, y = axial_to_xy(q, r, sp)
                test_positions.append((x, y))
            
            ang_list = [best_inner_rot] * len(test_positions) if 'best_inner_rot' in dir() else [0.0] * len(test_positions)
            
            # Check if all disjoint
            if check_all_disjoint(test_positions, ang_list):
                side = compute_required_side(test_positions, ang_list, (0.0, 0.0), best_outer_angle)
                if side < best_side:
                    best_side = side
                    best_config = (test_positions, ang_list)
    
    # Verify final configuration is valid
    if best_config is not None:
        final_positions, final_angles = best_config
        # Verify disjointness
        if not check_all_disjoint(final_positions, final_angles, eps=1e-6):
            # Fall back to original
            final_positions = [p for p in best_positions]
            final_angles = [0.0] * len(final_positions)
            best_side = compute_required_side(final_positions, final_angles, (0.0, 0.0), best_outer_angle)
    else:
        final_positions = [p for p in best_positions]
        final_angles = [0.0] * len(final_positions)
        best_side = compute_required_side(final_positions, final_angles, (0.0, 0.0), best_outer_angle)
    
    # Format inner hexagons as (x, y, angle_degrees)
    inner = [[p[0], p[1], a] for p, a in zip(final_positions, final_angles)]
    
    return inner, (0.0, 0.0), best_side, best_outer_angle
# EVOLVE-BLOCK-END
