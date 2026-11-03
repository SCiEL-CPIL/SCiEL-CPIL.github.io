"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill

    # Unit regular hexagon vertices (flat-top orientation, angle=0)
    # For a regular hexagon with side length s, the vertices are at distance s from center
    # Flat-top: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Pointy-top: vertices at angles 30, 90, 150, 210, 270, 330 degrees
    # Let's use flat-top: vertex i at angle i*60 degrees, radius = side_length
    hex_verts_0 = np.array([[math.cos(math.radians(60*i)), math.sin(math.radians(60*i))] for i in range(6)])
    
    def rotate_verts(verts, angle_deg):
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        return verts @ R.T
    
    def hex_verts(angle_deg):
        return rotate_verts(hex_verts_0, angle_deg)
    
    def min_enclosing_hexagon(points, outer_angle_deg=0.0):
        """Find minimum side length of a regular hexagon (with given rotation) enclosing all points."""
        # A regular hexagon with side length s and rotation theta can be described as:
        # intersection of 6 half-planes. For a flat-top hexagon centered at origin with side s:
        # The apothem (distance from center to side) = s * sqrt(3)/2
        # The 6 supporting lines are at distance apothem from center, with normals at angles
        # 0, 60, 120, 180, 240, 300 degrees for flat-top.
        # 
        # For a hexagon rotated by theta, the normals are at theta + i*60 for i=0..5.
        # The constraint is: for all points p, for all i, n_i . (p - center) <= apothem
        # where n_i are unit normals.
        # 
        # We want to minimize s (or equivalently apothem = s * sqrt(3)/2) over center (cx, cy).
        # 
        # Let a = apothem = s * sqrt(3)/2. Then s = 2a / sqrt(3).
        # Minimize a subject to: n_i . (p_j - c) <= a for all i, j
        # i.e., n_i . c <= n_i . p_j - a for all i, j
        # i.e., a + n_i . c >= n_i . p_j for all i, j
        # 
        # Variables: x = [a, cx, cy]
        # Minimize a (x[0])
        # Constraints: for each i, j: a + n_i_x * cx + n_i_y * cy >= n_i . p_j
        #   => -a - n_i_x * cx - n_i_y * cy <= -n_i . p_j
        # Also a >= 0, but that's implied if points are non-empty.
        
        theta = math.radians(outer_angle_deg)
        normals = np.array([[math.cos(theta + math.radians(60*i)), math.sin(theta + math.radians(60*i))] for i in range(6)])
        
        # Build LP
        n_pts = len(points)
        n_constraints = 6 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        for j, p in enumerate(points):
            for i in range(6):
                n = normals[i]
                # a + n[0]*cx + n[1]*cy >= n[0]*p[0] + n[1]*p[1]
                # => -a - n[0]*cx - n[1]*cy <= -(n[0]*p[0] + n[1]*p[1])
                A_ub[6*j + i] = [-1, -n[0], -n[1]]
                b_ub[6*j + i] = -(n[0]*p[0] + n[1]*p[1])
        
        c_obj = np.array([1.0, 0.0, 0.0])
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (None, None), (None, None)], method='highs')
        
        if res.status == 0:
            a_opt = res.x[0]
            s_opt = 2 * a_opt / math.sqrt(3)
            center = (res.x[1], res.x[2])
            return s_opt, center
        else:
            return float('inf'), (0.0, 0.0)
    
    def get_all_points(inner_hexes):
        """Get all vertices of all inner hexagons."""
        all_pts = []
        for (cx, cy, ang) in inner_hexes:
            verts = hex_verts(ang)
            all_pts.extend([verts[i] + np.array([cx, cy]) for i in range(6)])
        return np.array(all_pts)
    
    def check_disjoint(inner_hexes, eps=1e-9):
        """Check if inner hexagons are pairwise disjoint (no overlapping interiors).
        Two convex polygons are disjoint if no point of one is strictly inside the other
        and they don't share edges in a way that creates overlap.
        For regular hexagons, a simpler check: if the distance between centers is less than
        2*inradius and they overlap, that's bad. But rotation matters.
        Let's do a proper check: for each pair, check if any vertex of one is inside the other."""
        n = len(inner_hexes)
        for i in range(n):
            for j in range(i+1, n):
                ci, ri, ai = inner_hexes[i]
                cj, rj, aj = inner_hexes[j]
                vi = hex_verts(ai) + np.array(ci)
                vj = hex_verts(aj) + np.array(cj)
                # Check if any vertex of i is inside j, or any vertex of j is inside i
                if any(_point_in_hexagon(vi[k], cj, aj) for k in range(6)):
                    return False
                if any(_point_in_hexagon(vj[k], ci, ai) for k in range(6)):
                    return False
        return True
    
    def _point_in_hexagon(p, center, angle_deg):
        """Check if point p is inside or on boundary of hexagon at center with rotation angle_deg."""
        theta = math.radians(angle_deg)
        normals = np.array([[math.cos(theta + math.radians(60*i)), math.sin(theta + math.radians(60*i))] for i in range(6)])
        # For unit hexagon, apothem = sqrt(3)/2
        apothem = math.sqrt(3)/2
        rel = np.array(p) - np.array(center)
        for n in normals:
            if n[0]*rel[0] + n[1]*rel[1] > apothem + 1e-9:
                return False
        return True

    # Generate candidate configurations
    # Strategy: Try different arrangements of N unit hexagons
    # Common compact arrangements:
    # 1. Hexagonal packing (honeycomb-like)
    # 2. Linear arrangements
    # 3. Clustered arrangements
    
    candidates = []
    
    # Helper: generate honeycomb-like cluster
    def honeycomb_cluster(n_hex):
        """Generate a compact honeycomb-like cluster of n_hex unit hexagons."""
        # In a honeycomb lattice, adjacent hexagons have center distance = 2 * apothem = sqrt(3)
        # for flat-top orientation. Actually, for unit side length, the distance between
        # centers of adjacent hexagons in a honeycomb is 2 * (sqrt(3)/2) = sqrt(3) for 
        # flat-to-flat contact, or 2 for vertex-to-vertex.
        # For flat-top hexagons, the 6 neighbors are at distance sqrt(3) in directions 0,60,...
        # Wait, let me think. Flat-top hexagon with side 1:
        # Vertices at (1,0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1,0), (-1/2,-sqrt(3)/2), (1/2,-sqrt(3)/2)
        # The distance from center to a side (apothem) = sqrt(3)/2
        # Two hexagons sharing a flat side: center distance = 2 * apothem = sqrt(3)
        # The direction to the shared side normal is 0, 60, 120, etc.
        
        # Let's use a hexagonal lattice with spacing sqrt(3) in 6 directions
        # Directions: 0, 60, 120, 180, 240, 300 degrees, distance sqrt(3)
        
        if n_hex == 1:
            return [(0.0, 0.0, 0.0)]
        
        # Generate positions in hexagonal coordinates
        # Use axial coordinates: position = q * e1 + r * e2
        # where e1 = (sqrt(3)/2, 1/2) * 2 * (sqrt(3)/2) ... let me just use the 6 directions
        
        dir_vectors = []
        for i in range(6):
            a = math.radians(60 * i)
            dir_vectors.append((math.cos(a) * math.sqrt(3), math.sin(a) * math.sqrt(3)))
        
        # BFS to generate first n_hex positions in hexagonal lattice
        visited = set()
        queue = [(0, 0)]
        visited.add((0, 0))
        positions = []
        
        while queue and len(positions) < n_hex:
            q, r = queue.pop(0)
            positions.append((q, r))
            if len(positions) >= n_hex:
                break
            for di in range(6):
                nq = q + [1, 0, -1, -1, 0, 1][di]
                nr = r + [0, 1, 1, 0, -1, -1][di]
                if (nq, nr) not in visited:
                    visited.add((nq, nr))
                    queue.append((nq, nr))
        
        # Convert axial to Cartesian
        # e1 = dir_vectors[0], e2 = dir_vectors[1]
        e1 = np.array(dir_vectors[0])
        e2 = np.array(dir_vectors[1])
        
        result = []
        for (q, r) in positions[:n_hex]:
            pos = q * e1 + r * e2
            result.append((float(pos[0]), float(pos[1]), 0.0))
        
        return result
    
    # Candidate 1: Honeycomb cluster
    candidates.append(honeycomb_cluster(N))
    
    # Candidate 2: Linear row
    # For flat-top hexagons, placing in a row along x-axis:
    # Each hexagon has width 2 (from -1 to 1), so to tile them side by side with flat sides touching,
    # center distance = sqrt(3) in the perpendicular direction... 
    # Actually for a row, let's just space them by 2*sqrt(3)/2 * 2 = 2*sqrt(3) if sharing flat sides
    # No, let me just try spacing = sqrt(3) and see.
    spacing = math.sqrt(3)
    row = [(k * spacing, 0.0, 0.0) for k in range(-(N//2), N - N//2)]
    candidates.append(row)
    
    # Candidate 3: Two rows offset
    if N >= 2:
        row1_n = (N + 1) // 2
        row2_n = N - row1_n
        row1 = [((k - (row1_n-1)/2) * spacing, 0.0, 0.0) for k in range(row1_n)]
        row2 = [((k - (row2_n-1)/2) * spacing, spacing * math.sqrt(3)/2, 0.0) for k in range(row2_n)]
        candidates.append(row1 + row2)
    
    # Candidate 4: 3xN or similar grid
    # Try a 2-wide arrangement
    for width in [2, 3]:
        height = (N + width - 1) // width
        arr = []
        for h in range(height):
            for w in range(width):
                if len(arr) >= N:
                    break
                # Hexagonal grid: offset every other row
                x = w * spacing
                y = h * spacing * math.sqrt(3)
                if h % 2 == 1:
                    x += spacing / 2
                arr.append((x, y, 0.0))
        if len(arr) == N:
            candidates.append(arr)
    
    # Candidate 5: Compact blob - try a few different seed patterns
    # Diamond/rhombus shape
    for offset in [0.5, 1.0]:
        arr = []
        for d in range(-3, 4):
            for e in range(-3, 4):
                if abs(d) + abs(e) <= 3 and len(arr) < N:
                    x = d * spacing + e * spacing * 0.5
                    y = e * spacing * math.sqrt(3) / 2
                    arr.append((x, y, 0.0))
        if len(arr) == N:
            candidates.append(arr[:N])
    
    # Evaluate each candidate
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for idx, inner in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
        
        # First check disjointness
        if not check_disjoint(inner):
            continue
        
        # Try different outer hexagon rotations
        best_for_candidate = float('inf')
        best_center_for_candidate = (0.0, 0.0)
        
        for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
            if time.time() > deadline - 3.0:
                break
            
            pts = get_all_points(inner)
            s, center = min_enclosing_hexagon(pts, outer_angle)
            
            if s < best_for_candidate:
                best_for_candidate = s
                best_center_for_candidate = center
        
        if best_for_candidate < best_side:
            best_side = best_for_candidate
            best_inner = inner
            best_center = best_center_for_candidate
            best_angle = 0.0  # We need to track which angle gave the best
    
    # Refine: try to find the optimal outer angle for the best candidate
    if best_inner is not None and time.time() < deadline - 10.0:
        pts = get_all_points(best_inner)
        best_refined = best_side
        best_refined_center = best_center
        best_refined_angle = best_angle
        
        # Fine sweep of outer angles
        for outer_angle in np.linspace(0, 60, 25):
            if time.time() > deadline - 5.0:
                break
            s, center = min_enclosing_hexagon(pts, outer_angle)
            if s < best_refined:
                best_refined = s
                best_refined_center = center
                best_refined_angle = outer_angle
        
        best_side = best_refined
        best_center = best_refined_center
        best_angle = best_refined_angle
    
    # If no valid candidate found, fallback
    if best_inner is None:
        # Fallback: just place them in a row with large spacing
        inner = [(k * 10.0, 0.0, 0.0) for k in range(N)]
        pts = get_all_points(inner)
        s, center = min_enclosing_hexagon(pts, 0.0)
        return inner, center, s, 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
