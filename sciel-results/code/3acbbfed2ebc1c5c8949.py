import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(all_vertices, angle_deg):
    """
    For a fixed outer hexagon angle, find optimal center and side length using LP.
    Returns (cx, cy, side) or None on failure.
    """
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0

    A_ub = []
    b_ub = []

    # Outer hexagon normals for given angle
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            # (vx - cx)*nx + (vy - cy)*ny <= r  =>  -nx*cx - ny*cy - r <= -vx*nx - vy*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
            # -(vx - cx)*nx - (vy - cy)*ny <= r  =>  nx*cx + ny*cy - r <= vx*nx + vy*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(vx * nx + vy * ny)

    # r >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(None, None), (None, None), (0, None)],
                         method='highs', options={'presolve': True})
        if result.success:
            cx_opt = result.x[0]
            cy_opt = result.x[1]
            r_opt = result.x[2]
            side = r_opt / (math.sqrt(3) / 2)
            return cx_opt, cy_opt, side
    except Exception:
        pass
    return None

def _generate_lattice_points(S):
    """Generate lattice points for a hexagon of side S (radius S)."""
    sqrt3 = math.sqrt(3)
    points = []
    # Lattice basis:
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    # Point (i, j) -> x = i*sqrt(3) + j*sqrt(3)/2, y = j*1.5
    for i in range(-S, S + 1):
        for j in range(-S, S + 1):
            # Check if inside hexagon: |i| <= S, |j| <= S, |i+j| <= S
            if abs(i) <= S and abs(j) <= S and abs(i + j) <= S:
                x = i * sqrt3 + j * sqrt3 / 2
                y = j * 1.5
                points.append((x, y))
    return points

def _find_best_outer_for_config(all_vertices):
    """
    Find the best outer hexagon (center, side, angle) for a fixed set of inner vertices.
    Returns (cx, cy, side, angle_deg).
    """
    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)

    # Centroid for initial guess (not strictly needed for LP, but good for sanity)
    if not all_vertices:
        return 0.0, 0.0, 0.0, 0.0
    
    cx0 = sum(v[0] for v in all_vertices) / len(all_vertices)
    cy0 = sum(v[1] for v in all_vertices) / len(all_vertices)

    # Phase 1: Coarse sweep over outer angles (0 to 60, step 2)
    for angle_deg in np.arange(0, 60, 2.0):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    # Phase 2: Fine sweep around best angle (step 0.2)
    for angle_deg in np.arange(best_angle - 2.0, best_angle + 2.0, 0.2):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    # Phase 3: Very fine polish (step 0.02)
    for angle_deg in np.arange(best_angle - 0.2, best_angle + 0.2, 0.02):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        if time.time() > _DEADLINE - 5.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    # Phase 4: Final ultra-fine (step 0.002)
    for angle_deg in np.arange(best_angle - 0.02, best_angle + 0.02, 0.002):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        if time.time() > _DEADLINE - 3.0:
            break
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    sqrt3 = math.sqrt(3)
    
    # Generate lattice points for S=3 hexagon (19 points)
    points_S3 = _generate_lattice_points(3)
    
    # Separate into central (d<=1) and outer (d=2)
    # Distance in lattice coords: max(|i|, |j|, |i+j|)
    # We didn't store i, j, so we recalculate or just use Euclidean distance from origin as proxy?
    # Better: Regenerate with i, j to classify properly.
    
    points_with_coords = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            if abs(i) <= 3 and abs(j) <= 3 and abs(i + j) <= 3:
                x = i * sqrt3 + j * sqrt3 / 2
                y = j * 1.5
                dist = max(abs(i), abs(j), abs(i + j))
                points_with_coords.append((x, y, dist))
                
    central_points = [p for p in points_with_coords if p[2] <= 1]
    outer_points = [p for p in points_with_coords if p[2] == 2]
    
    # We expect 7 central and 12 outer
    # Construct base set: all central points
    base_positions = [(p[0], p[1]) for p in central_points]
    
    # We need to pick 4 from outer_points
    # To save time, we can prune some obviously bad combinations?
    # Or just run all 495.
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0

    # Try all combinations of 4 outer points
    # This is 495 combinations * 2 inner angles * LP sweep
    # 495 * 2 * ~50 LP calls = ~50,000 LP calls.
    # Each LP call is small (66 constraints, 3 vars). Should be fast.
    
    outer_positions_list = [(p[0], p[1]) for p in outer_points]
    
    for combo in itertools.combinations(range(len(outer_positions_list)), 4):
        if time.time() > _DEADLINE - 20.0:
            break
            
        selected_outer = [outer_positions_list[i] for i in combo]
        current_positions = base_positions + selected_outer
        
        # Try inner angles 0 and 30
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
                
            all_vertices = []
            for px, py in current_positions:
                verts = _get_hex_vertices(px, py, inner_angle)
                all_vertices.extend(verts)
            
            cx, cy, side, angle = _find_best_outer_for_config(all_vertices)
            
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in current_positions]
                best_center = (cx, cy)
                best_outer_angle = angle
                best_inner_angle = inner_angle

    if best_inner is None:
        # Fallback if optimization failed or no candidates
        # Use a simple line or grid
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
