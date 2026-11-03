"""
Optimized hexagon packing: 11 unit hexagons in a compact rigid hexagonal cluster.
Uses the 11 closest points on a triangular lattice, scaled slightly to ensure
strict disjointness if necessary, and optimizes the global rotation angle
relative to the outer container using Golden Section Search and Linear Programming.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin before 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """
    Get vertices of a unit regular hexagon centered at (cx, cy) with rotation.
    Standard orientation: flat top?
    Vertices at angles 0, 60, ... or 30, 90...?
    The problem statement says "regular hexagons".
    Usually, a hexagon with side 1 has vertices at distance 1 from center.
    If angle 0 is a vertex, vertices are at (1,0), (1/2, sqrt(3)/2), etc.
    If angle 0 is a flat side, vertices are at (sqrt(3)/2, 1/2), etc.
    The inspirations used:
    a = angle_rad + math.pi * k / 3
    verts.append((cx + math.cos(a), cy + math.sin(a)))
    This places a vertex at angle `angle_deg`.
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
    Returns (center_x, center_y, side_length).
    """
    # Normals of the outer hexagon sides.
    # If the outer hexagon has a vertex at angle `outer_angle_deg`,
    # its sides are perpendicular to directions `outer_angle_deg + 30 + 60*k`.
    # Wait, if a vertex is at theta, the adjacent sides are at theta+30 and theta-30.
    # The normal to the side between vertex theta and theta+60 is theta+30.
    # So normals are at outer_angle_deg + 30 + 60*k.
    
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

def _polish_outer(vertices, init_cx, init_cy, init_angle_deg, deadline):
    """
    Polish the outer hexagon parameters (cx, cy, angle) using Nelder-Mead.
    Objective: Side length of the enclosing hexagon.
    """
    def objective(params):
        cx, cy, angle_deg = params
        # We need to find the min R for this specific cx, cy, angle.
        # But _find_min_enclosing_hex_side optimizes cx, cy too.
        # To use Nelder-Mead effectively on (cx, cy, angle), we should fix cx, cy 
        # and just calculate the required R? 
        # No, if we fix cx, cy, the min R is max over all vertices and normals of (v-c).n.
        # This is a smooth (max of linear) function! Much better for NM.
        
        angle_rad = math.radians(angle_deg)
        normals = []
        for k in range(6):
            a = angle_rad + math.pi / 6 + math.pi * k / 3
            normals.append((math.cos(a), math.sin(a)))
            
        max_proj = 0.0
        for vx, vy in vertices:
            dx = vx - cx
            dy = vy - cy
            for nx, ny in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
                    # Early exit not possible easily as we need max over all
        # R = max_proj
        # Side = 2 * R / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    x0 = [init_cx, init_cy, init_angle_deg]
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 2000})
        if time.time() > deadline:
            return None
        if res.success:
            cx, cy, angle_deg = res.x
            # Verify with LP to get exact side
            final_res = _find_min_enclosing_hex_side(vertices, angle_deg)
            if final_res:
                fcx, fcy, fside = final_res
                # Check if the LP center is close to the NM center? 
                # Actually, LP finds the global min for that angle. 
                # NM found a good angle. 
                # We should return the LP result for that angle as it is exact for that angle.
                # But wait, NM optimized (cx, cy, angle). The LP for the optimal angle 
                # might find a different cx, cy that is even better.
                # So we should just return the LP result for the best angle found by NM.
                return fcx, fcy, fside, angle_deg
    except Exception:
        pass
    return None

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
    # Note: There might be ties. The sort is stable in Python 3, 
    # but the order of insertion depends on the loop.
    # To be robust, we can break ties by angle or just take the first 11.
    # The inspirations took the first 11 from a similar sort.
    # Let's verify the points selected.
    # (0,0): 0
    # (1,0): 3, (0,1): 3, (-1,1): 3, (-1,0): 3, (0,-1): 3, (1,-1): 3
    # (1,1): 9, (0,2): 12, (-1,2): 9, (-2,1): 9, (-2,0): 12, (-2,-1): 21, (-1,-1): 9, (0,-2): 12, (1,-2): 9, (2,-1): 9, (2,0): 12
    
    # The 7 inner points are clear.
    # The next 4 points will be chosen from the 6 points with d_sq=9.
    # The specific 4 chosen might matter for symmetry, but any 4 from the 6 
    # symmetric ring will likely yield the same minimal bounding hexagon 
    # if we allow rotation. However, the "shape" is different.
    # For 11 hexes, the optimal packing is known to be a specific cluster.
    # Let's just take the first 11 from the sorted list.
    
    selected = candidates[:N]
    
    # Convert axial to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    sqrt3 = math.sqrt(3)
    
    # Apply a very small scale to ensure strict disjointness if the evaluator is strict.
    # The inspirations used 1.000001. The current program used 1.001.
    # A smaller gap is better for score.
    # If "disjoint" means interiors are disjoint, touching is allowed.
    # If "disjoint" means sets are disjoint, touching is NOT allowed.
    # The task says "disjoint regular hexagons". In set theory, disjoint means intersection is empty.
    # Touching hexagons intersect at a boundary point.
    # So we MUST scale up slightly.
    scale = 1.000001
    
    positions = []
    for _, q, r in selected:
        x = 1.5 * q * scale
        y = sqrt3 * (r + q / 2.0) * scale
        positions.append((x, y))
    
    # The inner hexagons are all oriented at 0 degrees relative to the cluster.
    # We will rotate the entire cluster by `cluster_angle`.
    # The outer hexagon will be at `outer_angle`.
    # Relative angle is `outer_angle - cluster_angle`.
    # Since the cluster is rigid, we can fix `cluster_angle = 0` and just search `outer_angle`.
    # However, the "closest 11" cluster might not have 6-fold symmetry.
    # If we rotate the cluster, the set of vertices changes.
    # It is equivalent to rotating the outer hexagon relative to the fixed cluster.
    
    inner_hexes_base = [(x, y, 0.0) for x, y in positions]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Symmetry of the outer hexagon is 60 degrees.
    # We search in [0, 60) or [0, 30) if the cluster has 2-fold symmetry?
    # The 11-closest cluster (7 + 4 from 6) generally has no rotational symmetry.
    # So we must search the full 0-60 range.
    
    # Coarse search
    coarse_angles = np.arange(0, 60.0, 0.5)
    best_coarse_angle = 0.0
    best_coarse_side = float('inf')
    best_coarse_center = (0.0, 0.0)
    
    for ang_deg in coarse_angles:
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Get all vertices of the inner hexagons
        all_vertices = []
        for px, py, iang in inner_hexes_base:
            verts = _get_hex_vertices(px, py, iang)
            all_vertices.extend(verts)
        
        res = _find_min_enclosing_hex_side(all_vertices, ang_deg)
        if res:
            cx, cy, side = res
            if side < best_coarse_side:
                best_coarse_side = side
                best_coarse_angle = ang_deg
                best_coarse_center = (cx, cy)
    
    # Fine search around best coarse angle
    if best_coarse_side < float('inf'):
        search_range = 0.5
        step = 0.01
        fine_angles = np.arange(best_coarse_angle - search_range, best_coarse_angle + search_range + step, step)
        
        for ang_deg in fine_angles:
            if ang_deg < 0 or ang_deg >= 60:
                continue
            if time.time() > _DEADLINE - 5.0:
                break
            
            all_vertices = []
            for px, py, iang in inner_hexes_base:
                verts = _get_hex_vertices(px, py, iang)
                all_vertices.extend(verts)
            
            res = _find_min_enclosing_hex_side(all_vertices, ang_deg)
            if res:
                cx, cy, side = res
                if side < best_coarse_side:
                    best_coarse_side = side
                    best_coarse_angle = ang_deg
                    best_coarse_center = (cx, cy)
        
        best_side = best_coarse_side
        best_outer_angle = best_coarse_angle
        best_center = best_coarse_center
    
    if best_inner is None and best_side < float('inf'):
        best_inner = [[pos[0], pos[1], 0.0] for pos in positions]
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Polish the best solution found
    if time.time() < _DEADLINE - 2.0:
        all_vertices = []
        for px, py, iang in inner_hexes_base:
            verts = _get_hex_vertices(px, py, iang)
            all_vertices.extend(verts)
        
        polish_res = _polish_outer(all_vertices, best_center[0], best_center[1], best_outer_angle, _DEADLINE - 1.0)
        if polish_res:
            pcx, pcy, pside, pang = polish_res
            if pside < best_side:
                best_side = pside
                best_center = (pcx, pcy)
                best_outer_angle = pang

    return best_inner, best_center, best_side, best_outer_angle
