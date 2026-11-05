"""Stronger algorithm: grid-based placement with local refinement for unit hexagons in a hexagon."""
import numpy as np
import math
import time

N = 11  # number of unit hexagons

# Constants for unit regular hexagon (side length 1)
# A regular hexagon with side length s has:
# - circumradius (center to vertex) = s
# - inradius (center to edge) = s * sqrt(3)/2
# - width (vertex to vertex) = 2s
# - height (edge to edge) = s * sqrt(3)

S = 1.0  # side length of inner hexagons
CIRCUMRADIUS = S  # distance from center to vertex
INRADIUS = S * math.sqrt(3) / 2  # distance from center to edge midpoint

def hex_vertices(cx, cy, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with given rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * math.pi / 3
        x = cx + CIRCUMRADIUS * math.cos(a)
        y = cy + CIRCUMRADIUS * math.sin(a)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, angle_deg):
    """Check if point (px, py) is inside a regular hexagon centered at (cx, cy)."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = math.radians(angle_deg)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with circumradius R, a point (lx, ly) is inside if
    # max(|lx|, |ly| * 2/sqrt(3)) <= R * sqrt(3)/2 ... actually let me use a cleaner check
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # In local coords, the edges are at distances INRADIUS from center
    # The hexagon is the intersection of 6 half-planes
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    # For each edge, the constraint is: dot((lx,ly), normal) <= INRADIUS
    
    for i in range(6):
        a = math.pi / 6 + i * math.pi / 3  # 30, 90, 150, 210, 270, 330
        nx = math.cos(a)
        ny = math.sin(a)
        if lx * nx + ly * ny > INRADIUS + 1e-9:
            return False
    return True

def hexagons_disjoint(v1, v2, v3, v4, tol=1e-6):
    """Check if two hexagons (defined by their 6 vertices each) are disjoint (non-overlapping).
    Uses SAT (Separating Axis Theorem)."""
    # Get all edge normals from both hexagons
    axes = []
    for verts in [v1, v2]:
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            # Edge vector
            ex = x2 - x1
            ey = y2 - y1
            # Normal (perpendicular)
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx/norm, ny/norm))
    
    for nx, ny in axes:
        # Project both hexagons onto this axis
        proj1 = [x*nx + y*ny for x, y in v1]
        proj2 = [x*nx + y*ny for x, y in v2]
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        # Check for gap
        if max1 < min2 + tol or max2 < min1 + tol:
            return True  # Disjoint (separated by this axis)
    
    return False  # Overlapping

def check_all_disjoint(positions, angles, tol=1e-6):
    """Check that all hexagons are pairwise disjoint."""
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(hex_vertices(positions[i][0], positions[i][1], angles[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            if not hexagons_disjoint(verts_list[i], verts_list[j], tol):
                return False
    return True

def hexagons_inside_outer(positions, angles, outer_cx, outer_cy, outer_side, outer_angle):
    """Check that all inner hexagons are completely inside the outer hexagon."""
    # Outer hexagon vertices
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_angle)
    
    # For each inner hexagon, check all its vertices are inside the outer hexagon
    for i in range(len(positions)):
        inner_verts = hex_vertices(positions[i][0], positions[i][1], angles[i])
        for vx, vy in inner_verts:
            if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_angle):
                return False
    return True

def compute_min_outer_side(positions, angles, outer_cx=0.0, outer_cy=0.0, outer_angle=0.0):
    """Compute the minimum side length of an outer hexagon (centered at outer_cx, outer_cy with rotation outer_angle)
    that contains all inner hexagons."""
    # The outer hexagon must contain all vertices of all inner hexagons
    # For a hexagon centered at (cx, cy) with rotation angle, a point (px, py) is inside
    # if for all 6 edge normals: dot((px-cx, py-cy), normal_i) <= side * sqrt(3)/2
    
    # We want to find the minimum side length S_outer such that all points are inside
    # This is equivalent to: S_outer >= max over all points and all normals of dot(point - center, normal) / (sqrt(3)/2)
    
    max_val = 0.0
    angle = math.radians(outer_angle)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for i in range(len(positions)):
        verts = hex_vertices(positions[i][0], positions[i][1], angles[i])
        all_points.extend(verts)
    
    # For each of the 6 edge normals of the outer hexagon
    for i in range(6):
        a = math.pi / 6 + i * math.pi / 3
        nx = math.cos(a)
        ny = math.sin(a)
        
        for px, py in all_points:
            # Transform to outer hexagon's local frame
            dx = px - outer_cx
            dy = py - outer_cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            val = lx * nx + ly * ny
            if val > max_val:
                max_val = val
    
    # S_outer * sqrt(3)/2 >= max_val, so S_outer >= max_val * 2/sqrt(3)
    return max_val * 2.0 / math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second budget
    
    # Strategy: Try different grid arrangements and local refinements
    # The optimal packing of N unit hexagons in a hexagon is related to hexagonal lattice
    
    best_score = float('inf')
    best_positions = None
    best_angles = None
    
    # Initial guess: hexagonal lattice arrangement
    # For N hexagons, arrange in a hexagonal pattern
    # A hexagonal number arrangement: 1, 7, 19, 37, ...
    # For N=11, we can use a 3x3 grid with corners removed, or other patterns
    
    # Pattern 1: Linear arrangement (current seed)
    positions = []
    angles = []
    for k in range(N):
        positions.append([4.0 * k, 0.0])
        angles.append(0.0)
    
    # Compute score for this arrangement
    score = compute_min_outer_side(positions, angles)
    best_score = score
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return [[p[0], p[1], a] for p, a in zip(best_positions, best_angles)], (0.0, 0.0), best_score, 0.0
    
    # Try hexagonal lattice arrangements
    # For a hexagonal lattice, adjacent hexagons are at distance 2*INRADIUS = sqrt(3) apart
    # in the direction perpendicular to the shared edge
    
    # Generate several candidate patterns
    patterns = []
    
    # Pattern: 3-4-4 triangular arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons (but we only need 11 total)
    # Let's try: 3 + 4 + 4 = 11
    
    # Hexagonal lattice spacing
    # Two adjacent hexagons sharing an edge have centers at distance 2 * INRADIUS = sqrt(3)
    d = 2 * INRADIUS  # = sqrt(3)
    
    # Create a hexagonal lattice of positions
    # Using axial coordinates (q, r) for hex grid
    # Position in Cartesian: x = d * (q + r/2), y = d * r * sqrt(3)/2
    
    def hex_to_cart(q, r):
        x = d * (q + r / 2.0)
        y = d * r * math.sqrt(3) / 2.0
        return x, y
    
    # Generate candidate sets of hex grid positions
    candidate_sets = []
    
    # Set 1: Compact cluster
    # Center at (0,0), then neighbors
    # Axial coordinates for a compact cluster of 11
    # Ring 0: (0,0) -> 1
    # Ring 1: 6 neighbors -> 7 total
    # Ring 2: need 4 more from ring 2
    # Ring 2 has 12 positions
    ring0 = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    
    # Try different combinations of ring 2 positions
    from itertools import combinations
    for combo in combinations(range(12), 4):
        idxs = list(combo)
        set_pos = ring0 + ring1 + [ring2[i] for i in idxs]
        candidate_sets.append(set_pos)
    
    # Also try other arrangements
    # Linear: 11 in a row
    # 5-6 arrangement: two rows
    # 4-4-3 arrangement
    
    # Evaluate each candidate set
    for cset in candidate_sets:
        if time.time() > deadline - 15:
            break
        
        pos = []
        ang = [0.0] * N
        for q, r in cset:
            x, y = hex_to_cart(q, r)
            pos.append([x, y])
        
        # Check if all disjoint (they should be for proper hex lattice)
        if check_all_disjoint(pos, ang):
            s = compute_min_outer_side(pos, ang)
            if s < best_score:
                best_score = s
                best_positions = [p[:] for p in pos]
                best_angles = ang[:]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return [[p[0], p[1], a] for p, a in zip(best_positions, best_angles)], (0.0, 0.0), best_score, 0.0
    
    # Local refinement: try small perturbations of the best arrangement
    if best_positions is not None:
        # Try rotating individual hexagons
        for trial in range(200):
            if time.time() > deadline - 15:
                break
            
            # Random perturbation
            pos = [p[:] for p in best_positions]
            ang = best_angles[:]
            
            # Perturb one random hexagon
            idx = np.random.randint(0, N)
            pos[idx][0] += np.random.uniform(-0.5, 0.5)
            pos[idx][1] += np.random.uniform(-0.5, 0.5)
            ang[idx] = np.random.uniform(0, 60)
            
            if check_all_disjoint(pos, ang):
                s = compute_min_outer_side(pos, ang)
                if s < best_score:
                    best_score = s
                    best_positions = [p[:] for p in pos]
                    best_angles = ang[:]
    
    # Final: try to optimize outer hexagon center and angle
    if best_positions is not None:
        # Try different outer hexagon centers and angles
        for outer_cx in np.linspace(-1, 1, 5):
            for outer_cy in np.linspace(-1, 1, 5):
                for outer_angle in [0, 30, 60, 90, 120, 150]:
                    if time.time() > deadline - 15:
                        break
                    s = compute_min_outer_side(best_positions, best_angles, outer_cx, outer_cy, outer_angle)
                    if s < best_score:
                        best_score = s
    
    return [[p[0], p[1], a] for p, a in zip(best_positions, best_angles)], (0.0, 0.0), best_score, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Result: {len(result[0])} hexagons, outer side = {result[2]:.6f}")
