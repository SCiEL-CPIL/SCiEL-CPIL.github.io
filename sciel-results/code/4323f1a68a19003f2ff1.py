"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import linprog

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

def _min_enclosing_hex_side_lp(inner_hexes, angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices.
    Returns (center_x, center_y, side_length) or None.
    """
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    # Normals of the outer hexagon sides
    # Vertices of outer hex are at angle_deg + 60k
    # Normals are at angle_deg + 30 + 60k
    normals = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    # LP Variables: [cx, cy, R] where R is inradius
    # Minimize R
    c = [0.0, 0.0, 1.0]
    
    A_ub = []
    b_ub = []
    
    for vx, vy in all_vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            
            # (v - c) . n <= R  =>  -c . n - R <= -v . n
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-v_dot_n)
            
            # (c - v) . n <= R  =>   c . n - R <=  v . n
            A_ub.append([nx, ny, -1.0])
            b_ub.append(v_dot_n)
            
    # R >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
    except:
        pass
    return None

def _nelder_mead_polish(inner_hexes, center, side, angle, max_iter=100, tol=1e-8):
    """
    Polish the outer hexagon parameters (cx, cy, side, angle) using Nelder-Mead.
    We minimize side length subject to all inner vertices being inside the outer hex.
    Since LP gives exact side for fixed angle, we can just optimize angle and center via LP.
    But here we do a direct Nelder-Mead on (cx, cy, angle) minimizing the max violation or side.
    Actually, simpler: for a given angle, LP gives optimal (cx, cy, side). 
    So we just need to optimize angle. Let's do Nelder-Mead on angle only? 
    No, center also matters. But LP already optimizes center for fixed angle.
    So we only need to optimize angle. Let's do a 1D Nelder-Mead or just fine scan.
    The current code does coarse + fine scan. Let's add a Nelder-Mead polish on the 
    (cx, cy, angle) where we compute side via LP for each candidate.
    """
    from scipy.optimize import minimize
    
    def objective(params):
        cx, cy, angle_deg = params
        # Compute the required side length for this angle and center
        # We need to check if all vertices are inside, and find min side
        # Actually, for fixed center and angle, the min side is determined by the 
        # max over all vertices of the projection onto normals.
        normals = []
        for k in range(6):
            a = math.radians(angle_deg + 30 + 60 * k)
            normals.append((math.cos(a), math.sin(a)))
        
        all_vertices = []
        for ix, iy, iangle in inner_hexes:
            verts = _get_hex_vertices(ix, iy, iangle)
            all_vertices.extend(verts)
        
        max_R = 0
        for vx, vy in all_vertices:
            for nx, ny in normals:
                # (v - c) . n <= R
                val = (vx - cx) * nx + (vy - cy) * ny
                if val > max_R:
                    max_R = val
                # (c - v) . n <= R
                val2 = (cx - vx) * nx + (cy - vy) * ny
                if val2 > max_R:
                    max_R = val2
        
        R = max(0, max_R)
        side = R / (math.sqrt(3) / 2)
        return side
    
    x0 = [center[0], center[1], angle]
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': max_iter})
        if res.success:
            cx, cy, angle_deg = res.x
            # Get precise side from LP
            lp_res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
            if lp_res:
                return lp_res[0], lp_res[1], lp_res[2], angle_deg
        return center[0], center[1], side, angle
    except:
        return center[0], center[1], side, angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate 11 points on triangular lattice
    # Lattice vectors
    sqrt3 = math.sqrt(3)
    # Scale slightly to ensure disjointness
    scale = 1.0001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    candidates = []
    # Generate enough candidates to pick 11
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try inner angles 0 and 30 (symmetry of hexagon)
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 15.0:
            break
            
        inner_list = [[px, py, inner_angle] for px, py in positions]
        
        # Search for optimal outer angle
        # Due to symmetry, we only need to check 0 to 30 degrees
        # Coarse scan
        best_coarse_side = float('inf')
        best_coarse_angle = 0.0
        best_coarse_center = (0.0, 0.0)
        
        # Step 0.5 degrees
        for outer_angle_deg in np.arange(0, 30.1, 0.5):
            if time.time() > _DEADLINE - 15.0:
                break
            res = _min_enclosing_hex_side_lp(inner_list, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = outer_angle_deg
                    best_coarse_center = (cx, cy)
        
        # Fine scan around best coarse angle
        if best_coarse_angle > 0 or best_coarse_side < float('inf'):
            search_range = 0.5
            step = 0.01
            for outer_angle_deg in np.arange(max(0, best_coarse_angle - search_range), 
                                            min(30, best_coarse_angle + search_range) + step, step):
                if time.time() > _DEADLINE - 15.0:
                    break
                res = _min_enclosing_hex_side_lp(inner_list, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
        
        # Nelder-Mead polish on the best found configuration
        if best_coarse_side < float('inf') and time.time() < _DEADLINE - 20.0:
            polished = _nelder_mead_polish(inner_list, best_coarse_center, best_coarse_side, best_coarse_angle)
            if polished and polished[2] < best_coarse_side:
                best_coarse_center = (polished[0], polished[1])
                best_coarse_side = polished[2]
                best_coarse_angle = polished[3]
        
        if best_coarse_side < best_side:
            best_side = best_coarse_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_coarse_center
            best_outer_angle = best_coarse_angle
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Final polish with Nelder-Mead on the global best
    if time.time() < _DEADLINE - 30.0:
        inner_list = best_inner
        polished = _nelder_mead_polish(inner_list, best_center, best_side, best_outer_angle, max_iter=200, tol=1e-12)
        if polished and polished[2] < best_side:
            best_center = (polished[0], polished[1])
            best_side = polished[2]
            best_outer_angle = polished[3]

    return best_inner, best_center, best_side, best_outer_angle
