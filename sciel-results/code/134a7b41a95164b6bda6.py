import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = set()
    
    def add_candidate(axial_list):
        if len(axial_list) != N:
            return
        # Normalize: sort and remove duplicates
        sorted_axial = tuple(sorted(axial_list))
        candidates.add(sorted_axial)

    # Generate diverse topologies
    # 1. Compact hexagonal-like shapes
    # Center + 6 neighbors + 4 more
    add_candidate([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
                   (1,1), (-1,2), (-2,1), (2,-1)])
    add_candidate([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
                   (2,0), (-2,0), (0,2), (0,-2)])
    add_candidate([(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)])
    add_candidate([(0,0), (1,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)])
    add_candidate([(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)])
    add_candidate([(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (1,3)])
    add_candidate([(0,0), (1,0), (2,0), (3,0), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2)])
    add_candidate([(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3), (3,3)])
    
    # 2. Star / Radial shapes
    add_candidate([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
                   (2,0), (0,2), (-2,2), (-2,0), (0,-2)])
    add_candidate([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), 
                   (2,1), (1,2), (-1,2), (-2,1), (-1,-1)])

    # 3. Random walks (diverse structures)
    import random
    random.seed(42)
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for _ in range(1000):
        pts = [(0,0)]
        used = {(0,0)}
        for _ in range(N-1):
            base = random.choice(pts)
            dir = random.choice(neighbors)
            new_pt = (base[0] + dir[0], base[1] + dir[1])
            if new_pt not in used:
                pts.append(new_pt)
                used.add(new_pt)
        if len(pts) == N:
            add_candidate(pts)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def width_of_hexagon(pts, center, beta):
        # Calculate the width of the bounding hexagon of side length 1 centered at 'center'
        # that contains 'pts' when rotated by beta
        # The hexagon has apothem h = sqrt(3)/2.
        # The constraints are |dot(p - c, n_i)| <= h for 6 normal directions n_i
        
        # To be efficient, we compute the max projection
        max_proj = 0
        # Normals for a hexagon with side 1 rotated by beta
        # The normal directions are at angles beta + pi/6 + k*pi/3
        # Wait, for a regular hexagon with side length 1, the distance from center to side (apothem) is sqrt(3)/2.
        # The vertices are at distance 1.
        # The "width" in a direction perpendicular to a side is 2 * apothem = sqrt(3).
        # We want to find the minimal scaling factor S such that S * hexagon contains pts.
        # This is equivalent to finding min S such that for all normal directions n_k:
        # max_i dot(p_i - c, n_k) <= S * h AND min_i dot(p_i - c, n_k) >= -S * h
        # i.e., max_i |dot(p_i - c, n_k)| <= S * h
        # So S = max_k ( max_i |dot(p_i - c, n_k)| ) / h
        
        # We optimize S. Since S depends linearly on c and piecewise-linearly on beta, 
        # we can use numerical optimization.
        
        # Precompute points relative to center
        # For speed in optimization, we can't easily vectorize the max over i for all k easily without loops,
        # but N is small (11) and K is 6.
        
        rel_pts = pts - np.array(center)
        
        # Normal vectors
        # The sides of a unit hexagon are at distance h from center.
        # The normal to a side is at angle theta.
        # For a hexagon with one side horizontal (flat top/bottom), the normals are at 0, 60, 120, 180, 240, 300.
        # Let's define the hexagon such that its sides are perpendicular to the normals.
        # Standard orientation: vertex at (1, 0). Normals at 30, 90, 150...
        # Let's use the property that the width of a unit hexagon in any direction n is:
        # w(n) = 2 * h * max(1, |cos(angle_diff)| / cos(pi/6)) ... no, simpler:
        # The support function h_K(u) of a regular hexagon of side 1 (apothem h) is:
        # h_K(u) = h / cos(delta) where delta is the angle between u and the closest normal to a side.
        # Wait, we are fitting the POINTS inside a hexagon.
        # Let the hexagon be H = { x | |dot(x, n_k)| <= h for k=1..6 }.
        # We want min S such that pts subset of S * H_centered.
        # S * H = { x | |dot(x, n_k)| <= S * h }.
        # So we need max_i |dot(p_i - c, n_k)| <= S * h for all k.
        # S = max_k ( max_i |dot(p_i - c, n_k)| ) / h
        
        # The normals n_k depend on the orientation beta.
        # Let n_k(beta) = (cos(beta + k*pi/3 + pi/6), sin(beta + k*pi/3 + pi/6))
        # Note: The "sides" of the hexagon are perpendicular to these normals.
        
        best_max = 0
        for k in range(6):
            ang = beta + k * math.pi / 3.0 + math.pi / 6.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            # Project points onto this normal
            projs = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
            max_abs = np.max(np.abs(projs))
            if max_abs > best_max:
                best_max = max_abs
                
        return best_max / APOTHEM

    def optimize_shape(pts):
        # We want to find c (2D) and beta (scalar) to minimize S
        # S(c, beta) = max_k max_i |dot(p_i - c, n_k(beta))| / h
        
        # Initialize with centroid and beta=0
        c0 = np.mean(pts, axis=0)
        b0 = 0.0
        
        # Coarse search for beta to get a good start
        best_S = float('inf')
        best_b = 0.0
        best_c = c0
        
        for i in range(36):
            b = i * (math.pi / 6.0) / 35.0
            # For a fixed beta, the optimal c is the center of the minimal bounding box in the transformed space?
            # Actually, for fixed beta, the objective is convex in c.
            # We can minimize over c using a simple method or just evaluate at centroid first.
            # But c affects the max projection.
            # Let's just do a full Nelder-Mead on (cx, cy, beta)
            
        # Use Nelder-Mead for (cx, cy, beta)
        def objective(params):
            cx, cy, b = params
            # Normalize beta to [0, pi/6) to reduce landscape? 
            # Actually the function has period pi/6.
            return width_of_hexagon(pts, [cx, cy], b)
        
        x0 = [c0[0], c0[1], 0.0]
        
        # Try a few starting points for beta to escape local minima
        best_res = None
        best_fun = float('inf')
        
        for b_start in [0.0, math.pi/36.0, math.pi/12.0, math.pi/18.0, math.pi/9.0]:
            x0_try = [c0[0], c0[1], b_start]
            try:
                res = minimize(objective, x0_try, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                pass

        if best_res is None:
            return float('inf'), 0.0, (c0[0], c0[1])
            
        cx, cy, beta = best_res.x
        return best_fun, beta, (cx, cy)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Iterate through candidates
    # To save time, we limit the number of candidates if there are too many
    candidate_list = list(candidates)
    
    # If we have too many, subsample or prioritize compact ones?
    # For now, just iterate.
    
    for axial in candidate_list:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        axial = list(candidates)[0]
        pts = axial_to_cartesian(axial)
        best_global_pts = pts
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Output
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

if __name__ == "__main__":
    inner, center, side, rot = construct()
    print(f"Side: {side}, Center: {center}, Rot: {rot}")
    print(f"Inner count: {len(inner)}")
