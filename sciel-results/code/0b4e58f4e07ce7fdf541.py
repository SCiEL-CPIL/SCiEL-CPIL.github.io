"""
Optimized hexagon packing: 11 unit hexagons in a compact rigid hexagonal cluster.
Combines the robust LP-based outer hexagon sizing from the current program
with the lattice generation and angle exploration strategy from inspirations.
Implements a hierarchical search (coarse to fine) for the optimal relative
rotation between the inner cluster and the outer container.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin before 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) with rotation.
    Standard orientation: vertex at angle 0.
    Vertices are at angles angle_deg, angle_deg + 60, ...
    """
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _find_min_enclosing_hex_side(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length) or None if infeasible.
    """
    # Normals of the outer hexagon sides.
    # If the outer hexagon has a vertex at angle `outer_angle_deg`,
    # the normals to the sides are at `outer_angle_deg + 30 + 60*k`.
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # LP Variables: [cx, cy, R] where R is the apothem (inradius) of the outer hexagon.
    # Minimize R
    # Constraints for each vertex v and normal n:
    # (v - c) . n <= R  =>  -c . n - R <= -v . n
    # (c - v) . n <= R  =>   c . n - R <=  v . n
    
    A_ub = []
    b_ub = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # Constraint 1: -nx*cx - ny*cy - R <= -v_dot_n
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-v_dot_n)
            
            # Constraint 2: nx*cx + ny*cy - R <= v_dot_n
            A_ub.append([nx, ny, -1.0])
            b_ub.append(v_dot_n)
    
    # R >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    c_vec = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, 
                      bounds=[(None, None), (None, None), (0, None)], 
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            # Side length = 2 * apothem / sqrt(3)
            side = R * 2.0 / math.sqrt(3)
            return cx, cy, side
        else:
            return None
    except Exception:
        return None

def _smooth_enclosing_hex(vertices, outer_angle_deg, temperature=0.01):
    """
    Smooth surrogate for the enclosing hexagon optimization using log-sum-exp.
    Returns (center_x, center_y, smooth_side_estimate).
    """
    # Normals of the outer hexagon sides
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # For a given center (cx, cy) and apothem R, the constraint is:
    # max over all (v, n) of [(v - c) . n] <= R
    # We use a smooth approximation: R_smooth = temperature * log(sum(exp((v-c).n / temperature)))
    
    # This is still an optimization problem. Let's use a different approach:
    # We'll define a smooth objective that approximates the LP.
    
    # Actually, let's just use the LP but with a finer search.
    # The smooth surrogate approach would require solving a nonlinear program.
    
    # For now, return the LP result as the "smooth" estimate
    return _find_min_enclosing_hex_side(vertices, outer_angle_deg)

def _polish_angle_lp(vertices, best_angle, best_side, search_range=0.5, fine_step=0.001):
    """
    High-precision polish pass around the best angle found.
    Uses very fine steps to find the optimal angle.
    """
    best_cx, best_cy, best_s = None, None, best_side
    
    # Fine search around best angle
    fine_angles = np.arange(best_angle - search_range, best_angle + search_range + fine_step, fine_step)
    
    for ang_deg in fine_angles:
        if time.time() > _DEADLINE - 3.0:
            break
        
        # Wrap angle to [0, 60) if needed
        ang_wrapped = ang_deg % 60.0
        if ang_wrapped < 0:
            ang_wrapped += 60.0
        
        res = _find_min_enclosing_hex_side(vertices, ang_wrapped)
        if res:
            cx, cy, side = res
            if side < best_s:
                best_s = side
                best_cx, best_cy = cx, cy
                best_angle = ang_wrapped
    
    return best_cx, best_cy, best_s, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback: return a valid, albeit large, configuration
        # Just place them in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # 1. Generate the 11 closest points on a triangular lattice
    # Axial coordinates (q, r)
    # Distance squared from origin: 3 * (q^2 + r^2 + q*r)
    # We want the 11 smallest distances.
    
    candidates = []
    # Search range sufficient to find 11 points
    for q in range(-3, 4):
        for r in range(-3, 4):
            d_sq = 3 * (q*q + r*r + q*r)
            candidates.append((d_sq, q, r))
    
    candidates.sort(key=lambda x: x[0])
    
    # Take the 11 closest
    selected = candidates[:N]
    
    # Convert axial to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    sqrt3 = math.sqrt(3)
    
    # Apply a very small scale to ensure strict disjointness if the evaluator is strict.
    # "Disjoint" typically implies no intersection, so touching is not allowed.
    # A tiny scale factor ensures a microscopic gap.
    scale = 1.000001
    
    positions = []
    for _, q, r in selected:
        x = 1.5 * q * scale
        y = sqrt3 * (r + q / 2.0) * scale
        positions.append((x, y))
    
    # The inner hexagons are all oriented at 0 degrees relative to the cluster.
    # We will rotate the entire cluster by `cluster_angle` (implicitly 0 by fixing positions)
    # and search for the optimal `outer_angle`.
    # Since the cluster is rigid, rotating the cluster is equivalent to rotating the outer hexagon.
    
    inner_hexes_base = [(x, y, 0.0) for x, y in positions]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # The 11-closest cluster (7 center + 4 from the ring of 6) has no 6-fold symmetry.
    # It has a specific shape. We must search the full 0-60 degree range for the outer angle.
    
    # Pre-compute vertices for the inner hexagons (they don't change during angle search)
    all_vertices = []
    for px, py, iang in inner_hexes_base:
        verts = _get_hex_vertices(px, py, iang)
        all_vertices.extend(verts)
    
    # Coarse search with smaller step for better initial estimate
    coarse_step = 0.2
    coarse_angles = np.arange(0, 60.0, coarse_step)
    best_coarse_angle = 0.0
    best_coarse_side = float('inf')
    best_coarse_center = (0.0, 0.0)
    
    for ang_deg in coarse_angles:
        if time.time() > _DEADLINE - 15.0:
            break
        
        res = _find_min_enclosing_hex_side(all_vertices, ang_deg)
        if res:
            cx, cy, side = res
            if side < best_coarse_side:
                best_coarse_side = side
                best_coarse_angle = ang_deg
                best_coarse_center = (cx, cy)
    
    # Fine search around best coarse angle
    if best_coarse_side < float('inf'):
        search_range = coarse_step
        fine_step = 0.005
        fine_angles = np.arange(best_coarse_angle - search_range, best_coarse_angle + search_range + fine_step, fine_step)
        
        for ang_deg in fine_angles:
            # Wrap angle to [0, 60) if needed
            ang_wrapped = ang_deg % 60.0
            if ang_wrapped < 0:
                ang_wrapped += 60.0
            
            if time.time() > _DEADLINE - 8.0:
                break
            
            res = _find_min_enclosing_hex_side(all_vertices, ang_wrapped)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = ang_wrapped
                    best_coarse_center = (cx, cy)
        
        best_side = best_coarse_side
        best_outer_angle = best_coarse_angle
        best_center = best_coarse_center
        
        # High-precision polish pass
        if time.time() < _DEADLINE - 5.0:
            best_cx, best_cy, best_s, best_ang = _polish_angle_lp(
                all_vertices, best_outer_angle, best_side, 
                search_range=0.1, fine_step=0.0005
            )
            if best_cx is not None and best_s < best_side:
                best_side = best_s
                best_outer_angle = best_ang
                best_center = (best_cx, best_cy)
    
    if best_inner is None and best_side < float('inf'):
        best_inner = [[pos[0], pos[1], 0.0] for pos in positions]
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
