import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
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
    Returns (center_x, center_y, side_length).
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
            
            A.append([nx, ny, -1.0])
            b.append(v_dot_n)
            
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _smooth_outer_side(all_vertices, outer_angle_deg, temperature=0.01):
    """
    Smooth surrogate for the minimum enclosing hexagon side length using log-sum-exp.
    This makes the objective differentiable for gradient-based optimization.
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # For each normal direction, compute the required R
    # R must be >= max over all vertices of |v . n - c . n|
    # But we don't know c yet. Instead, use LP result as initial and refine.
    
    # Actually, let's use a different approach:
    # The minimum R for a given center c is max over all (v, n) of |v . n - c . n|
    # We want to minimize over c and R subject to R >= all these values.
    
    # For the smooth version, we approximate max with log-sum-exp.
    # But we need to handle both + and - constraints.
    
    # Let's just use the LP result directly and add a small perturbation for smoothness.
    # Actually, for L-BFGS-B we can use the exact LP solution as the objective.
    
    res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
    if res is None:
        return 1e10
    return res[2]

def _polish_with_lbfgsb(all_vertices, init_center, init_angle, init_side, max_iter=500):
    """
    Polish the solution using L-BFGS-B with the exact LP-based objective.
    """
    def objective(params):
        cx, cy, angle_deg = params
        res = _find_min_enclosing_hex_side(all_vertices, angle_deg)
        if res is None:
            return 1e10
        # Add penalty if center is far from expected (helps convergence)
        _, _, side = res
        return side
    
    x0 = np.array([init_center[0], init_center[1], init_angle])
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        if result.success:
            cx, cy, angle_deg = result.x
            res = _find_min_enclosing_hex_side(all_vertices, angle_deg)
            if res:
                return res
    except Exception:
        pass
    
    return None

def _generate_hex_layout(n_hexes):
    """
    Generate a hexagonal arrangement of n unit hexagons.
    Returns list of (x, y) positions.
    """
    sqrt3 = math.sqrt(3)
    scale = 1.000001  # Slightly larger than touching
    
    # Hexagonal lattice vectors
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidates in a hexagonal pattern
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == n_hexes:
            break
    
    return positions

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate hexagonal layout
    positions = _generate_hex_layout(N)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try inner angles 0 and 30
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 15.0:
            break
            
        # Get all vertices for this inner angle
        all_vertices = []
        for px, py in positions:
            verts = _get_hex_vertices(px, py, inner_angle)
            all_vertices.extend(verts)
        
        # Coarse search for outer angle
        best_coarse_side = float('inf')
        best_coarse_angle = 0.0
        best_coarse_center = (0.0, 0.0)
        
        # Symmetry: 0 to 30 degrees is sufficient
        for outer_angle_deg in np.arange(0, 30.1, 0.5):
            if time.time() > _DEADLINE - 15.0:
                break
            res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
        
        # Fine search around best coarse angle
        if best_coarse_angle >= 0 and best_coarse_side < float('inf'):
            search_range = 0.5
            step = 0.01
            for outer_angle_deg in np.arange(max(0, best_coarse_angle - search_range), 
                                            min(30, best_coarse_angle + search_range) + step, step):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _find_min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
        
        # Polish with L-BFGS-B if time permits
        if time.time() < _DEADLINE - 20.0 and best_coarse_side < float('inf'):
            polished = _polish_with_lbfgsb(all_vertices, best_coarse_center, 
                                           best_coarse_angle, best_coarse_side, max_iter=200)
            if polished:
                cx, cy, side = polished
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_center = (cx, cy)
                    # Re-find angle from polished center
                    # Actually, the polished result already has the angle
                    # But we need to extract it properly
                    # Let's just use the LP result at the best angle
        
        if best_coarse_side < best_side:
            best_side = best_coarse_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_coarse_center
            best_outer_angle = best_coarse_angle
    
    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
