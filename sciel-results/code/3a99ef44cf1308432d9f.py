import time
import math
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11

# --- Geometry helpers ---

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with side length 'side' and rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def hexagon_constraints(cx, cy, side, angle_deg):
    """
    For a regular hexagon centered at (cx,cy) with side s and rotation angle_deg,
    return the 6 linear constraints (n_i, b_i) such that n_i · p + b_i >= 0 for all p inside.
    Actually we need: for a point p to be inside, n_i · p <= b_i for each face.
    Let's define: the hexagon is the intersection of 6 half-planes.
    The normal to face i points outward. The constraint is: n_i · (p - c) <= inradius * side? 
    No: for a regular hexagon, the distance from center to each edge (apothem) = side * sqrt(3)/2.
    So constraint: n_i · (p - c) <= apothem, where n_i is unit outward normal to edge i.
    """
    apothem = side * math.sqrt(3) / 2
    a = np.radians(angle_deg)
    normals = []
    rhs = []
    for k in range(6):
        # Outward normal to edge k
        theta = a + k * np.pi / 3 + np.pi / 6  # edge normal is offset by 30 degrees from vertex direction
        # Actually, let's think: vertex k is at angle a + k*60. The edge between vertex k and k+1
        # has normal at angle a + k*60 + 30.
        nx = math.cos(a + k * np.pi / 3 + np.pi / 6)
        ny = math.sin(a + k * np.pi / 3 + np.pi / 6)
        normals.append([nx, ny])
        # Constraint: nx*(x-cx) + ny*(y-cy) <= apothem
        # => nx*x + ny*y <= apothem + nx*cx + ny*cy
        b = apothem + nx * cx + ny * cy
        rhs.append(b)
    return np.array(normals), np.array(rhs)

def min_enclosing_hexagon_side(points, angle_deg, center=None):
    """
    Given points and a fixed rotation angle for the outer hexagon, find the minimum side length
    such that all points are inside a hexagon with that angle and some center.
    
    We can optimize over the center as well. But for a given angle, the problem is:
    find center c and minimum side s such that for each of the 6 faces:
    n_i · (p_j - c) <= s * sqrt(3)/2 for all j
    => n_i · p_j - n_i · c <= s * sqrt(3)/2
    => n_i · c - n_i · p_j >= -s * sqrt(3)/2
    
    This is an LP in variables c_x, c_y, s.
    """
    from scipy.optimize import linprog
    
    pts = np.array(points)
    normals, _ = hexagon_constraints(0, 0, 1.0, angle_deg)
    
    # Variables: [cx, cy, s]
    # Constraints: for each face i and point j:
    # n_i · p_j - n_i · c - s * sqrt(3)/2 <= 0
    # => -n_i[0]*cx - n_i[1]*cy - (sqrt(3)/2)*s + n_i · p_j <= 0
    # A_ub @ vars <= b_ub where A_ub row: [-n_i0, -n_i1, -sqrt(3)/2], b_ub: n_i · p_j
    
    n_faces = 6
    n_pts = len(pts)
    A_ub = np.zeros((n_faces * n_pts, 3))
    b_ub = np.zeros(n_faces * n_pts)
    
    sqrt3_2 = math.sqrt(3) / 2
    for i in range(n_faces):
        ni = normals[i]
        for j in range(n_pts):
            idx = i * n_pts + j
            A_ub[idx, 0] = -ni[0]
            A_ub[idx, 1] = -ni[1]
            A_ub[idx, 2] = -sqrt3_2
            b_ub[idx] = ni[0] * pts[j, 0] + ni[1] * pts[j, 1]
    
    # Objective: minimize s => c = [0, 0, 1]
    c_obj = np.array([0.0, 0.0, 1.0])
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0.5, None)], method='highs')
        if res.success:
            return res.fun
    except:
        pass
    
    # Fallback: use bounding box heuristic
    # Project points onto the 6 normals, find range, then side = max_range / sqrt(3)
    best_s = float('inf')
    for trial_cx in [0.0]:
        for trial_cy in [0.0]:
            s_max = 0.0
            for i in range(6):
                ni = normals[i]
                proj = pts[:, 0] * ni[0] + pts[:, 1] * ni[1]
                # Max deviation from center
                dev = proj - (trial_cx * ni[0] + trial_cy * ni[1])
                s_max = max(s_max, np.max(np.abs(dev)))
            s_candidate = s_max / sqrt3_2
            if s_candidate < best_s:
                best_s = s_candidate
    return best_s

def find_best_configuration(centers, deadline):
    """
    Given a set of inner hexagon centers, find the best outer hexagon (center, side, angle).
    Returns (side, angle, center) tuple or None.
    """
    pts = np.array(centers)
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Coarse grid over angles
    for ang in range(0, 180, 5):
        if time.time() > deadline - 10:
            break
        s = min_enclosing_hexagon_side(pts, ang)
        if s < best_side:
            best_side = s
            best_angle = ang
    
    # Refine with golden section
    if time.time() < deadline - 5:
        try:
            result = minimize_scalar(lambda a: min_enclosing_hexagon_side(pts, a), 
                                    bounds=(best_angle - 5, best_angle + 5), 
                                    method='bounded', options={'xatol': 1e-6})
            if result.success and result.fun < best_side:
                best_side = result.fun
                best_angle = result.x
        except:
            pass
    
    # Find optimal center for the best angle
    # Re-solve LP to get center
    from scipy.optimize import linprog
    normals, _ = hexagon_constraints(0, 0, 1.0, best_angle)
    sqrt3_2 = math.sqrt(3) / 2
    n_faces = 6
    n_pts = len(pts)
    A_ub = np.zeros((n_faces * n_pts, 3))
    b_ub = np.zeros(n_faces * n_pts)
    for i in range(n_faces):
        ni = normals[i]
        for j in range(n_pts):
            idx = i * n_pts + j
            A_ub[idx, 0] = -ni[0]
            A_ub[idx, 1] = -ni[1]
            A_ub[idx, 2] = -sqrt3_2
            b_ub[idx] = ni[0] * pts[j, 0] + ni[1] * pts[j, 1]
    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0.5, None)], method='highs')
        if res.success:
            best_center = (res.x[0], res.x[1])
            best_side = res.fun
    except:
        pass
    
    return best_side, best_angle, best_center

def hexagonal_lattice_positions(rows, cols, spacing):
    """Generate positions for a hexagonal lattice arrangement."""
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = j * spacing * math.sqrt(3) + (i % 2) * spacing * math.sqrt(3) / 2
            y = i * spacing * 1.5
            positions.append([x, y])
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # --- Generate candidate configurations ---
    candidates = []
    
    # Candidate 1: Hexagonal lattice, centered
    # For N=11, we can try 3-4-3-1 or 4-3-4 or 3-4-4 etc.
    # Standard hex packing: rows of 3,4,3,1 = 11 or 4,3,4 = 11
    row_configs = [
        [3, 4, 3, 1],
        [4, 3, 4],
        [3, 3, 3, 2],
        [2, 3, 3, 3],
        [1, 3, 4, 3],
    ]
    
    for row_cfg in row_configs:
        if sum(row_cfg) != N:
            continue
        # Build positions
        spacing = math.sqrt(3)  # min distance between centers for unit hexagons
        total_height = (len(row_cfg) - 1) * spacing * 1.5
        positions = []
        y_offset = -total_height / 2
        for i, count in enumerate(row_cfg):
            y = y_offset + i * spacing * 1.5
            row_width = (count - 1) * spacing * math.sqrt(3)
            x_offset = -row_width / 2
            for j in range(count):
                x = x_offset + j * spacing * math.sqrt(3)
                positions.append([x, y])
        candidates.append(positions)
    
    # Candidate 2: Linear arrangement
    spacing = math.sqrt(3)
    linear_pos = [[k * spacing, 0.0] for k in range(N)]
    # Center it
    cx = np.mean([p[0] for p in linear_pos])
    cy = np.mean([p[1] for p in linear_pos])
    linear_pos = [[p[0] - cx, p[1] - cy] for p in linear_pos]
    candidates.append(linear_pos)
    
    # Candidate 3: Two rows
    # 6 on top, 5 on bottom
    spacing = math.sqrt(3)
    two_row = []
    for j in range(6):
        two_row.append([(j - 2.5) * spacing * math.sqrt(3), spacing * 1.5 / 2])
    for j in range(5):
        two_row.append([(j - 2.0) * spacing * math.sqrt(3), -spacing * 1.5 / 2])
    candidates.append(two_row)
    
    # Candidate 4: Three rows 4-3-4
    spacing = math.sqrt(3)
    three_row = []
    for j in range(4):
        three_row.append([(j - 1.5) * spacing * math.sqrt(3), -spacing * 1.5])
    for j in range(3):
        three_row.append([(j - 1.0) * spacing * math.sqrt(3), 0.0])
    for j in range(4):
        three_row.append([(j - 1.5) * spacing * math.sqrt(3), spacing * 1.5])
    candidates.append(three_row)
    
    # Candidate 5: Try some random perturbations of the best lattice
    if time.time() < deadline - 20:
        np.random.seed(42)
        base = candidates[0]  # first hex lattice
        for trial in range(5):
            perturbed = [[p[0] + np.random.uniform(-0.1, 0.1), p[1] + np.random.uniform(-0.1, 0.1)] for p in base]
            candidates.append(perturbed)
    
    # --- Evaluate each candidate ---
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    best_centers = None
    
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        
        result = find_best_configuration(cand, deadline)
        if result is not None:
            s, ang, ctr = result
            if s < best_side:
                best_side = s
                best_angle = ang
                best_center = ctr
                best_centers = cand
    
    # Fallback if nothing worked
    if best_centers is None:
        spacing = math.sqrt(3)
        inner = [[k * spacing, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    # Construct inner hexagons: all axis-aligned (angle 0)
    inner = [[x, y, 0.0] for x, y in best_centers]
    
    return inner, (best_center[0], best_center[1]), best_side, best_angle

# EVOLVE-BLOCK-START
# (The construct function above is the main logic)
# EVOLVE-BLOCK-END
