"""Hexagon packing via hex-lattice enumeration + analytic minimal enclosing hexagon."""
import numpy as np
import time
from scipy.optimize import linprog

N = 11
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, R, angle_deg):
    """Vertices of regular hexagon (circumradius R) centered at (cx,cy), rotated angle_deg."""
    a = np.deg2rad(angle_deg)
    th = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + R * np.cos(th), cy + R * R * np.sin(th)])


def _hex_face_normals(angle_deg):
    """6 outward unit normals of a regular hexagon rotated by angle_deg.
    For angle=0 (pointy-top, vertices at 0,60,...), face i (between vert i and i+1)
    has outward normal at angle (i+0.5)*60deg."""
    a = np.deg2rad(angle_deg)
    return np.column_stack([
        np.cos(a + (np.arange(6) + 0.5) * np.pi / 3.0),
        np.sin(a + (np.arange(6) + 0.5) * np.pi / 3.0)
    ])


def _hex_face_offsets(angle_deg, R):
    """Distance from center to each face (apothem = R * sqrt(3)/2)."""
    return np.full(6, R * SQRT3 / 2.0)


def _min_enclosing_hexagon(points, angle_deg):
    """Find minimum circumradius R and center (cx,cy) of a regular hexagon (rotated angle_deg)
    containing all points. Returns (R, cx, cy) or None if infeasible."""
    normals = _hex_face_normals(angle_deg)  # (6,2)
    offsets = _hex_face_offsets(angle_deg, 1.0)  # all = sqrt(3)/2 * R, so use unit offsets
    # For each face k: n_k . (p - c) <= R * sqrt(3)/2
    # => n_k . c - n_k . p_k <= R * sqrt(3)/2
    # Variables: x = [cx, cy, R]
    # Constraint: n_k . [cx,cy] - n_k . p + (sqrt(3)/2)*R >= 0 for all k, all points
    # => -n_k . [cx,cy] + n_k . p - (sqrt(3)/2)*R <= 0
    # Plus R >= 0
    m = len(points)
    A = np.zeros((6 * m + 1, 3))
    b = np.zeros(6 * m + 1)
    for k in range(6):
        nk = normals[k]
        for i in range(m):
            row = 6 * i + k
            A[row, 0] = -nk[0]
            A[row, 1] = -nk[1]
            A[row, 2] = SQRT3 / 2.0
            b[row] = nk[0] * points[i, 0] + nk[1] * points[i, 1]
    # R >= 0 => -R <= 0
    A[-1, 2] = -1.0
    b[-1] = 0.0
    c_obj = np.array([0.0, 0.0, 1.0])
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)],
                  method='highs')
    if res.status != 0:
        return None
    R = res.x[2]
    cx, cy = res.x[0], res.x[1]
    return R, cx, cy


def _hex_lattice_positions(radius):
    """Generate positions of hexagons on a hex lattice with axial coords up to given radius.
    Returns list of (x, y) for each hex center."""
    positions = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) > radius:
                continue
            # Convert axial to cartesian (pointy-top hex lattice)
            x = SQRT3 * (q + r / 2.0)
            y = 1.5 * r
            positions.append((x, y))
    return positions


def _all_subsets_of_size(n, k):
    """Generate all combinations of indices 0..n-1 taken k at a time."""
    import itertools
    return list(itertools.combinations(range(n), k))


def _validate_packing(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check that all inner unit hexagons are inside outer hexagon and don't overlap."""
    # Check containment: each inner vertex must be inside outer hexagon
    for (cx, cy) in inner_positions:
        vx, vy = _hex_vertices(cx, cy, 1.0, 0.0)
        for i in range(6):
            if not _point_in_hex(vx[i], vy[i], outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    # Check pairwise non-overlap: for lattice positions with distance >= 2, they don't overlap
    # (unit hexagons on hex lattice are tangent, not overlapping)
    for i in range(len(inner_positions)):
        for j in range(i + 1, len(inner_positions)):
            dx = inner_positions[i][0] - inner_positions[j][0]
            dy = inner_positions[i][1] - inner_positions[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 2.0 - tol:
                return False
    return True


def _point_in_hex(px, py, cx, cy, R, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy) with circumradius R."""
    normals = _hex_face_normals(angle_deg)
    offsets = R * SQRT3 / 2.0
    dx = px - cx
    dy = py - cy
    for k in range(6):
        if normals[k, 0] * dx + normals[k, 1] * dy > offsets + 1e-9:
            return False
    return True


def _min_distance_between_hexes(p1, p2):
    """Minimum distance between two unit hexagons at positions p1, p2 (both angle 0).
    For hex lattice, if center distance >= 2, they don't overlap.
    Return negative if overlapping."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    d = np.sqrt(dx * dx + dy * dy)
    # For two unit hexagons (circumradius 1) on hex lattice, min distance = 2.0
    # If d < 2, they overlap
    return d - 2.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    best_side = 1e18
    best_result = None

    # Generate hex lattice positions up to radius 2 (gives 19 positions)
    lattice_pos = _hex_lattice_positions(2)
    n_lattice = len(lattice_pos)
    print(f"Lattice positions: {n_lattice}")

    # Generate all C(19,11) = 75582 subsets
    subsets = _all_subsets_of_size(n_lattice, N)
    print(f"Subsets to check: {len(subsets)}")

    # For each subset, try multiple outer hexagon angles and find min enclosing
    # The outer hexagon angle is the key continuous parameter
    # Try angles: 0, 30 (and we'll do golden-section on the best few)
    
    # First pass: coarse angle search
    candidate_angles = np.arange(0, 30, 1.0)  # 0 to 29 degrees (period is 60, but 30 is enough due to symmetry)
    
    angle_sides = {}  # angle -> (min_side, subset_idx, center)
    
    for angle_deg in candidate_angles:
        if time.time() > deadline - 20:
            break
        min_side_for_angle = 1e18
        best_subset_for_angle = None
        best_center_for_angle = None
        
        for si, subset in enumerate(subsets):
            if time.time() > deadline - 20:
                break
            pts = np.array([lattice_pos[i] for i in subset])
            res = _min_enclosing_hexagon(pts, angle_deg)
            if res is None:
                continue
            R, cx, cy = res
            if R < min_side_for_angle:
                min_side_for_angle = R
                best_subset_for_angle = subset
                best_center_for_angle = (cx, cy)
        
        if best_subset_for_angle is not None:
            angle_sides[angle_deg] = (min_side_for_angle, best_subset_for_angle, best_center_for_angle)
            print(f"Angle {angle_deg:.1f}: side={min_side_for_angle:.6f}")

    if not angle_sides:
        # Fallback
        inner = [[SQRT3 * i, 0.0, 0.0] for i in range(-5, 6)]
        return inner, (0.0, 0.0), 50.0, 0.0

    # Find the best angle from coarse search
    best_angle_coarse = min(angle_sides.keys(), key=lambda a: angle_sides[a][0])
    best_side_coarse, best_subset_coarse, best_center_coarse = angle_sides[best_angle_coarse]
    print(f"Best coarse: angle={best_angle_coarse:.1f}, side={best_side_coarse:.6f}")

    # Second pass: golden-section search around best angle
    if time.time() < deadline - 10:
        # Search in [best_angle - 5, best_angle + 5]
        lo = max(0.0, best_angle_coarse - 5.0)
        hi = min(60.0, best_angle_coarse + 5.0)
        
        def eval_angle(a):
            if time.time() > deadline - 10:
                return 1e18
            min_side = 1e18
            for si, subset in enumerate(subsets):
                if time.time() > deadline - 10:
                    break
                pts = np.array([lattice_pos[i] for i in subset])
                res = _min_enclosing_hexagon(pts, a)
                if res is None:
                    continue
                R, cx, cy = res
                if R < min_side:
                    min_side = R
            return min_side

        # Golden section
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        f1 = eval_angle(x1)
        f2 = eval_angle(x2)
        
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            if f1 < f2:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - gr * (hi - lo)
                f1 = eval_angle(x1)
            else:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + gr * (hi - lo)
                f2 = eval_angle(x2)
        
        best_angle_fine = (lo + hi) / 2.0
        best_side_fine = min(f1, f2)
        print(f"Best fine: angle={best_angle_fine:.4f}, side={best_side_fine:.6f}")
        
        if best_side_fine < best_side_coarse:
            best_angle_coarse = best_angle_fine
            best_side_coarse = best_side_fine

    # Final: recompute best subset and center at the best angle
    final_angle = best_angle_coarse
    min_side = 1e18
    final_subset = None
    final_center = None
    
    for si, subset in enumerate(subsets):
        if time.time() > deadline - 5:
            break
        pts = np.array([lattice_pos[i] for i in subset])
        res = _min_enclosing_hexagon(pts, final_angle)
        if res is None:
            continue
        R, cx, cy = res
        if R < min_side:
            min_side = R
            final_subset = subset
            final_center = (cx, cy)
    
    if final_subset is None:
        # Fallback
        inner = [[SQRT3 * i, 0.0, 0.0] for i in range(-5, 6)]
        return inner, (0.0, 0.0), 50.0, 0.0

    # Build inner positions
    inner_positions = [lattice_pos[i] for i in final_subset]
    outer_cx, outer_cy = final_center
    outer_side = min_side
    outer_angle = final_angle

    # Validate
    if not _validate_packing(inner_positions, (outer_cx, outer_cy), outer_side, outer_angle):
        # Try to fix: slightly enlarge
        outer_side *= 1.01
        if not _validate_packing(inner_positions, (outer_cx, outer_cy), outer_side, outer_angle):
            # Fallback
            inner = [[SQRT3 * i, 0.0, 0.0] for i in range(-5, 6)]
            return inner, (0.0, 0.0), 50.0, 0.0

    # Format output
    inner = [[float(x), float(y), 0.0] for (x, y) in inner_positions]
    return inner, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
