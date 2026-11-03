"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses lattice-shape enumeration + fast 2D optimization of outer hexagon placement.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 135.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True


def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def hex_lattice_coord_to_xy(q, r, scale=2.0):
    """Convert axial hex coordinates to Cartesian. scale = distance between adjacent centers."""
    x = scale * (q + 0.5 * r)
    y = scale * (math.sqrt(3.0) / 2.0 * r)
    return (x, y)


def generate_all_12_cell_shapes(max_radius=4):
    """Generate all connected 12-cell polyhexes on the hex lattice within max_radius from origin."""
    # Use backtracking to generate all connected sets of 12 cells
    # Start from a base cell and grow
    all_shapes = set()
    
    def add_shape(cells):
        """Add shape to set with canonical form (normalized)."""
        # Normalize: translate so min-q, min-r cell is at (0,0)
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        norm = frozenset((c[0] - min_q, c[1] - min_r) for c in cells)
        # Also try all 6 rotations to get canonical form
        canonical = None
        for rot in range(6):
            rotated = set()
            for (q, r) in norm:
                # Rotate by 60 degrees * rot
                if rot == 0:
                    rotated.add((q, r))
                elif rot == 1:
                    rotated.add((-r, q + r))
                elif rot == 2:
                    rotated.add((-q - r, q))
                elif rot == 3:
                    rotated.add((-q, -r))
                elif rot == 4:
                    rotated.add((r, -q - r))
                elif rot == 5:
                    rotated.add((q + r, -q))
            # Normalize again
            min_q2 = min(c[0] for c in rotated)
            min_r2 = min(c[1] for c in rotated)
            norm2 = frozenset((c[0] - min_q2, c[1] - min_r2) for c in rotated)
            if canonical is None or norm2 < canonical:
                canonical = norm2
        all_shapes.add(canonical)
    
    def backtrack(cells, available):
        if len(cells) == 12:
            add_shape(list(cells))
            return
        if time.time() > DEADLINE - 30:
            return
        # Try adding each available cell
        for cell in list(available):
            cells.append(cell)
            # Update available: add neighbors of new cell that aren't in cells
            q, r = cell
            neighbors = [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
            for nb in neighbors:
                if nb not in cells and abs(nb[0]) <= max_radius and abs(nb[1]) <= max_radius:
                    available.add(nb)
            backtrack(cells, available)
            cells.pop()
            # Remove cells that were only reachable through this cell
            for nb in neighbors:
                if nb not in cells:
                    # Check if nb is still connected to cells
                    # Simple check: if nb has no neighbor in cells, remove from available
                    has_conn = False
                    for (dq, dr) in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
                        if (nb[0]+dq, nb[1]+dr) in cells:
                            has_conn = True
                            break
                    if not has_conn:
                        available.discard(nb)
    
    # Start with cell (0,0)
    backtrack([(0, 0)], {(0, 0)})
    
    return list(all_shapes)


def shape_to_positions(shape, scale=2.0, inner_angle=0.0):
    """Convert a canonical shape (set of axial coords) to Cartesian positions."""
    positions = []
    for (q, r) in sorted(shape):
        x, y = hex_lattice_coord_to_xy(q, r, scale)
        positions.append((x, y, inner_angle))
    return positions


def fast_optimize_outer(inner_positions, n_angles=24):
    """For a fixed set of inner positions, optimize outer center and angle.
    Returns best (outer_center, outer_side, outer_angle)."""
    # Precompute all vertices of all inner hexagons
    all_verts = []
    for (x, y, ang) in inner_positions:
        p = hex_vertices(x, y, 1.0, ang)
        all_verts.extend(p)
    all_verts = np.array(all_verts)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different outer angles
    for outer_angle in np.linspace(0, 60, n_angles, endpoint=False):
        a = math.radians(outer_angle)
        normals = np.array([(math.cos(a + (i + 0.5) * math.pi / 3.0), 
                            math.sin(a + (i + 0.5) * math.pi / 3.0)) for i in range(6)])
        
        # For each normal direction, the constraint is:
        # max over all vertices of (v - c) · n <= apothem
        # where apothem = s * sqrt(3)/2
        # So: c · n >= max(v · n) - apothem
        # And: c · n <= min(v · n) + apothem
        
        # We want to find c that satisfies all 12 constraints (6 normals, 2 signs each)
        # and minimizes the required apothem (hence side length)
        
        # For a given angle, the minimum enclosing hexagon side is determined by
        # the maximum of: for each normal n, (max(v·n) - min(v·n)) / 2 + |c·n - centroid·n|
        # Actually, let's think differently.
        
        # The outer hexagon with center c and angle a contains point v iff:
        # |v - c| projected onto each of the 6 face normals <= apothem
        # i.e., for each normal n_i: (v - c) · n_i <= apothem AND -(v - c) · n_i <= apothem
        # i.e., c · n_i >= v · n_i - apothem AND c · n_i <= v · n_i + apothem
        
        # For all vertices v and all normals n_i:
        # max_v (v · n_i) - apothem <= c · n_i <= min_v (v · n_i) + apothem
        
        # So we need: max_v(v·n_i) - min_v(v·n_i) <= 2 * apothem
        # i.e., apothem >= (max_v(v·n_i) - min_v(v·n_i)) / 2
        
        # And c must satisfy: for each i, c · n_i in [max_v(v·n_i) - apothem, min_v(v·n_i) + apothem]
        
        # The minimum apothem for this angle is:
        # apothem_min = max over i of (max_v(v·n_i) - min_v(v·n_i)) / 2
        
        # But we also need to check that there exists a c satisfying all constraints.
        # For a regular hexagon, if the width in each of the 6 directions is <= 2*apothem,
        # then a hexagon of that apothem centered appropriately will contain all points.
        
        # The optimal center c is the centroid-like point that minimizes the maximum
        # distance to the constraint boundaries.
        
        # For now, let's just compute the required apothem and use the centroid as center.
        # Then refine with a small local optimization.
        
        apothem_required = 0.0
        for i in range(6):
            n = normals[i]
            projections = all_verts @ n
            width = np.max(projections) - np.min(projections)
            apothem_required = max(apothem_required, width / 2.0)
        
        if apothem_required <= 0:
            continue
        
        # Side length from apothem: s = 2 * apothem / sqrt(3)
        side = 2.0 * apothem_required / math.sqrt(3.0)
        
        # Find optimal center: minimize max_i |c·n_i - mid_i| where mid_i = (max_v(v·n_i) + min_v(v·n_i))/2
        # This is a linear programming problem, but for 6 constraints in 2D, we can use
        # the fact that the optimal c is at the intersection of constraint boundaries.
        
        # Simple approach: use the average of the midpoints as initial center, then verify
        midpoints = []
        for i in range(6):
            n = normals[i]
            projections = all_verts @ n
            mid = (np.max(projections) + np.min(projections)) / 2.0
            midpoints.append(mid)
        
        # Solve for c using least squares on the 6 constraints c·n_i = mid_i
        # This is overdetermined (6 equations, 2 unknowns), so use lstsq
        A = normals  # 6x2
        b = np.array(midpoints)
        c_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        c_opt = (c_sol[0], c_sol[1])
        
        # Verify this center works
        max_proj = 0.0
        for (px, py) in all_verts:
            dx, dy = px - c_opt[0], py - c_opt[1]
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
        
        actual_apothem = max_proj
        actual_side = 2.0 * actual_apothem / math.sqrt(3.0)
        
        if actual_side < best_side:
            best_side = actual_side
            best_center = c_opt
            best_angle = outer_angle
    
    return best_center, best_side, best_angle


def construct():
    t0 = time.time()
    
    # Generate all 12-cell polyhex shapes
    shapes = generate_all_12_cell_shapes(max_radius=4)
    
    if time.time() > DEADLINE - 10:
        # Fallback
        positions = [(0, 0)]
        for k in range(6):
            theta = math.radians(k * 60.0)
            positions.append((2.0 * math.cos(theta), 2.0 * math.sin(theta)))
        for k in range(5):
            theta = math.radians(k * 60.0)
            positions.append((4.0 * math.cos(theta), 4.0 * math.sin(theta)))
        inner = [(x, y, 0.0) for (x, y) in positions]
        outer_side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
        return inner, (0.0, 0.0), outer_side, 0.0
    
    best_side = float('inf')
    best_result = None
    
    # Sort shapes by their bounding box size as a heuristic
    def shape_size(shape):
        xs = [hex_lattice_coord_to_xy(q, r, 2.0)[0] for (q, r) in shape]
        ys = [hex_lattice_coord_to_xy(q, r, 2.0)[1] for (q, r) in shape]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))
    
    shapes.sort(key=shape_size)
    
    # Try each shape
    for shape in shapes:
        if time.time() > DEADLINE - 15:
            break
        
        positions = shape_to_positions(shape, scale=2.0, inner_angle=0.0)
        
        # Try a few inner rotation angles
        for inner_rot in [0.0, 30.0]:
            if time.time() > DEADLINE - 15:
                break
            
            rot_positions = [(x, y, inner_rot) for (x, y, _) in positions]
            
            center, side, angle = fast_optimize_outer(rot_positions, n_angles=12)
            
            if side < best_side:
                # Verify validity
                if check_validity(rot_positions, center, side, angle):
                    best_side = side
                    best_result = (rot_positions, center, side, angle)
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple hexagonal cluster
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * math.cos(theta), 2.0 * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((4.0 * math.cos(theta), 4.0 * math.sin(theta)))
    inner = [(x, y, 0.0) for (x, y) in positions]
    outer_side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
    return inner, (0.0, 0.0), outer_side, 0.0
