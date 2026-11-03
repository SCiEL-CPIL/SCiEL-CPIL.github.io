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

def _quick_min_side(inner_hexes, angle_deg):
    """Quick computation of min enclosing hex side for given angle, optimizing center analytically."""
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    normals = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    max_R = 0.0
    for vx, vy in all_vertices:
        for nx, ny in normals:
            val = abs(vx * nx + vy * ny)
            if val > max_R:
                max_R = val
    
    side = max_R / (math.sqrt(3) / 2)
    return side

def _nelder_mead_polish(inner_hexes, center, side, angle, max_iter=100, tol=1e-8):
    """
    Polish the outer hexagon parameters (cx, cy, side, angle) using Nelder-Mead.
    """
    from scipy.optimize import minimize
    
    def objective(params):
        cx, cy, angle_deg = params
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
                val = (vx - cx) * nx + (vy - cy) * ny
                if val > max_R:
                    max_R = val
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
            lp_res = _min_enclosing_hex_side_lp(inner_hexes, angle_deg)
            if lp_res:
                return lp_res[0], lp_res[1], lp_res[2], angle_deg
        return center[0], center[1], side, angle
    except:
        return center[0], center[1], side, angle

def _generate_lattice_clusters():
    """
    Generate candidate cluster topologies on the hexagonal lattice.
    Each cluster is a set of 11 lattice positions that form a connected, 
    non-overlapping arrangement of unit hexagons.
    Returns list of position lists (each position is (x, y)).
    """
    sqrt3 = math.sqrt(3)
    scale = 1.0001  # Slight scale to ensure disjointness
    
    # Lattice basis vectors for hexagonal packing
    # For unit hexagons, centers are spaced by sqrt(3) in one direction
    # and (sqrt(3)/2, 1.5) in another
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate a large grid of candidate positions
    grid = {}
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            grid[(i, j)] = (x, y)
    
    # Define several known good cluster shapes for 11 hexagons
    # Shape 1: Compact hexagonal cluster (center + 6 + 4 partial)
    # A perfect hexagonal number pattern: 1 + 6 + 12 = 19 is too many
    # So we take subsets
    
    clusters = []
    
    # Shape 1: Row of 3, then 2 rows of 4, then row of 3? No, let's use standard patterns
    # Pattern: centered hexagon of side 2 has 19 cells. We need 11.
    
    # Pattern A: 3-4-3-1 arrangement (roughly)
    pattern_a = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # 5 in a row
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),  # 5 in next row
        (2, 2)                                      # 1 in third row
    ]
    
    # Pattern B: More compact 3-4-3-1
    pattern_b = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    
    # Pattern C: Diamond-like
    pattern_c = [
        (2, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    
    # Pattern D: 2-3-4-2
    pattern_d = [
        (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    
    # Pattern E: 4-3-4
    pattern_e = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    
    # Pattern F: 5-3-3 (staggered)
    pattern_f = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    
    # Pattern G: 2-4-3-2
    pattern_g = [
        (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    
    # Pattern H: 3-5-3
    pattern_h = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    
    # Pattern I: 1-5-5 (two rows of 5, one on top)
    pattern_i = [
        (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    
    # Pattern J: 2-5-4
    pattern_j = [
        (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    
    all_patterns = [pattern_a, pattern_b, pattern_c, pattern_d, pattern_e, 
                    pattern_f, pattern_g, pattern_h, pattern_i, pattern_j]
    
    for pattern in all_patterns:
        positions = []
        valid = True
        for (i, j) in pattern:
            if (i, j) in grid:
                positions.append(grid[(i, j)])
            else:
                valid = False
                break
        if valid and len(positions) == 11:
            clusters.append(positions)
    
    # Also try the original compact cluster from the current program
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    candidates.sort(key=lambda t: t[0])
    compact_positions = [(x, y) for _, x, y in candidates[:11]]
    clusters.append(compact_positions)
    
    return clusters

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Generate multiple cluster topologies
    clusters = _generate_lattice_clusters()
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Try each cluster topology with different inner angles
    for positions in clusters:
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Try inner angles 0 and 30 (symmetry of hexagon)
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
                
            inner_list = [[px, py, inner_angle] for px, py in positions]
            
            # Quick scan to find promising outer angles
            best_quick_side = float('inf')
            best_quick_angle = 0.0
            
            # Coarse scan with step 1 degree
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 10.0:
                    break
                side = _quick_min_side(inner_list, outer_angle_deg)
                if side < best_quick_side:
                    best_quick_side = side
                    best_quick_angle = outer_angle_deg
            
            # Fine scan around best angle
            if best_quick_side < float('inf'):
                for outer_angle_deg in np.arange(max(0, best_quick_angle - 1.0), 
                                                min(30, best_quick_angle + 1.0) + 0.05, 0.05):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    side = _quick_min_side(inner_list, outer_angle_deg)
                    if side < best_quick_side:
                        best_quick_side = side
                        best_quick_angle = outer_angle_deg
            
            # If this looks promising, do precise LP optimization
            if best_quick_side < best_side + 0.1:  # Only refine if competitive
                if time.time() > _DEADLINE - 10.0:
                    break
                
                best_lp_side = float('inf')
                best_lp_angle = best_quick_angle
                best_lp_center = (0.0, 0.0)
                
                # LP-based fine scan
                for outer_angle_deg in np.arange(max(0, best_quick_angle - 1.0), 
                                                min(30, best_quick_angle + 1.0) + 0.01, 0.01):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side_lp(inner_list, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_lp_side:
                            best_lp_side = side
                            best_lp_angle = outer_angle_deg
                            best_lp_center = (cx, cy)
                
                # Nelder-Mead polish
                if best_lp_side < float('inf') and time.time() < _DEADLINE - 15.0:
                    polished = _nelder_mead_polish(inner_list, best_lp_center, best_lp_side, best_lp_angle)
                    if polished and polished[2] < best_lp_side:
                        best_lp_center = (polished[0], polished[1])
                        best_lp_side = polished[2]
                        best_lp_angle = polished[3]
                
                if best_lp_side < best_side:
                    best_side = best_lp_side
                    best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                    best_center = best_lp_center
                    best_outer_angle = best_lp_angle
    
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Final polish with Nelder-Mead on the global best
    if time.time() < _DEADLINE - 20.0:
        inner_list = best_inner
        polished = _nelder_mead_polish(inner_list, best_center, best_side, best_outer_angle, max_iter=200, tol=1e-12)
        if polished and polished[2] < best_side:
            best_center = (polished[0], polished[1])
            best_side = polished[2]
            best_outer_angle = polished[3]

    return best_inner, best_center, best_side, best_outer_angle
