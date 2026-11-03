import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _hex_vertices(angle):
    # 6 vertices of a unit regular hexagon (side=1) rotated by 'angle' radians
    # Vertices at distance 1 from center, starting at angle
    return np.array([[math.cos(angle + k * math.pi / 3.0),
                      math.sin(angle + k * math.pi / 3.0)] for k in range(6)])

def _hex_halfspace_normals(angle):
    # 6 outward normals of a unit regular hexagon (side=1) rotated by 'angle'
    # Normal directions are at angle + pi/6 + k*pi/3
    return np.array([[math.cos(angle + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(angle + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _pairwise_overlap_penalty(V1, N1, V2, N2):
    # Compute minimum L1 sum of violation depths for two convex hexagons
    # Hex1 with vertices V1 (6x2) and outward normals N1 (6x2)
    # Hex2 with vertices V2 (6x2) and outward normals N2 (6x2)
    # For each normal of hex1, check max projection of hex2 vertices
    # If max_proj > 1 (apothem), violation = max_proj - 1
    # Similarly for hex2 normals vs hex1 vertices
    # Sum all positive violations
    # This is a convex lower bound on the actual separation needed
    
    # Project hex2 vertices onto hex1 normals
    proj2_on_1 = V2 @ N1.T  # (6, 6)
    max_proj2_on_1 = np.max(proj2_on_1, axis=0)  # (6,)
    viol_1 = np.maximum(0.0, max_proj2_on_1 - 1.0)  # apothem = 1 for unit hex? No, apothem = sqrt(3)/2
    
    # Wait, for a unit hexagon (side=1), the apothem is sqrt(3)/2.
    # The support function in normal direction n is max dot(v, n) = apothem.
    # So if max_proj > apothem, they overlap.
    viol_1 = np.maximum(0.0, max_proj2_on_1 - APOTHEM)
    
    # Project hex1 vertices onto hex2 normals
    proj1_on_2 = V1 @ N2.T  # (6, 6)
    max_proj1_on_2 = np.max(proj1_on_2, axis=0)  # (6,)
    viol_2 = np.maximum(0.0, max_proj1_on_2 - APOTHEM)
    
    return np.sum(viol_1) + np.sum(viol_2)

def _compute_objective(params):
    # params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, cx, cy, S, theta]
    # N=11, so 3*11 = 33 for inner, +4 for outer = 37 total
    inner_params = params[:3 * N]
    cx = params[3 * N]
    cy = params[3 * N + 1]
    S = params[3 * N + 2]
    theta = params[3 * N + 3]
    
    penalty = 0.0
    
    # Compute all inner hexagon vertices and normals
    V_list = []
    N_list = []
    for i in range(N):
        xi = inner_params[3 * i]
        yi = inner_params[3 * i + 1]
        ai = inner_params[3 * i + 2]
        verts = _hex_vertices(ai) + np.array([xi, yi])
        norms = _hex_halfspace_normals(ai)
        V_list.append(verts)
        N_list.append(norms)
    
    # Pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            penalty += _pairwise_overlap_penalty(V_list[i], N_list[i], V_list[j], N_list[j])
    
    # Containment: each inner hexagon must be inside the outer hexagon
    # Outer hexagon: center (cx, cy), side S, rotation theta
    # Outer apothem = S * APOTHEM
    # Outer normals: theta + pi/6 + k*pi/3
    outer_norms = np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                             math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    outer_apothem = S * APOTHEM
    
    for i in range(N):
        # Project inner hexagon vertices onto outer normals
        # Inner vertices relative to outer center
        V_rel = V_list[i] - np.array([cx, cy])
        proj = V_rel @ outer_norms.T  # (6, 6)
        max_proj = np.max(proj, axis=0)  # (6,)
        # If max_proj > outer_apothem, violation
        viol = np.maximum(0.0, max_proj - outer_apothem)
        penalty += np.sum(viol)
    
    # Also add a small regularization to prefer smaller S
    penalty += 0.1 * S
    
    return penalty

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a valid configuration
        # Simple grid of 11 hexagons
        pts = []
        # 3x3 grid minus 2 corners + 2 extra
        for q in range(-1, 2):
            for r in range(-1, 2):
                if len(pts) < 11:
                    pts.append((q, r))
        # Convert axial to cartesian
        cartesian = []
        for q, r in pts:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            cartesian.append((x, y, 0.0))
        
        # Compute bounding hexagon
        cx = float(np.mean([p[0] for p in cartesian]))
        cy = float(np.mean([p[1] for p in cartesian]))
        
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            for x, y, _ in cartesian:
                proj = (x - cx) * nx + (y - cy) * ny
                max_proj = max(max_proj, abs(proj))
        
        S = (max_proj + APOTHEM) / APOTHEM
        
        inner = [[float(x), float(y), 0.0] for x, y, _ in cartesian]
        return inner, [cx, cy], S, 0.0
    
    # Initial guess: compact arrangement
    # Use a hexagonal grid of 11 cells
    initial_axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)]
    initial_cartesian = []
    for q, r in initial_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        initial_cartesian.append((x, y, 0.0))
    
    # Estimate initial outer size
    cx0 = float(np.mean([p[0] for p in initial_cartesian]))
    cy0 = float(np.mean([p[1] for p in initial_cartesian]))
    max_proj0 = 0.0
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        for x, y, _ in initial_cartesian:
            proj = (x - cx0) * nx + (y - cy0) * ny
            max_proj0 = max(max_proj0, abs(proj))
    S0 = (max_proj0 + APOTHEM) / APOTHEM
    
    # Build initial params
    init_params = []
    for x, y, a in initial_cartesian:
        init_params.extend([x, y, a])
    init_params.extend([cx0, cy0, S0, 0.0])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-15, 15))  # x
        bounds.append((-15, 15))  # y
        bounds.append((0, 2 * math.pi))  # angle
    bounds.append((-15, 15))  # cx
    bounds.append((-15, 15))  # cy
    bounds.append((3, 15))  # S (minimum ~3 for 11 hexagons)
    bounds.append((0, math.pi / 3))  # theta (6-fold symmetry)
    
    # Run differential evolution
    best_result = None
    best_score = float('inf')
    
    try:
        def _obj_wrapper(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e10
            try:
                return _compute_objective(params)
            except Exception:
                return 1e10
        
        result = differential_evolution(
            _obj_wrapper,
            bounds=bounds,
            x0=init_params,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.9,
            polish=False,  # Disable polish to save time
            init='latinhypercube'
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
    except Exception:
        pass
    
    # If DE didn't find anything good, use initial
    if best_result is None:
        best_result = np.array(init_params)
    
    # Extract solution
    inner_params = best_result[:3 * N]
    cx = best_result[3 * N]
    cy = best_result[3 * N + 1]
    S = best_result[3 * N + 2]
    theta = best_result[3 * N + 3]
    
    # Verify and adjust S to be exactly the minimum needed
    # Recompute the exact S needed for the found positions and angles
    V_list = []
    for i in range(N):
        xi = inner_params[3 * i]
        yi = inner_params[3 * i + 1]
        ai = inner_params[3 * i + 2]
        verts = _hex_vertices(ai) + np.array([xi, yi])
        V_list.append(verts)
    
    # Find minimum S for given theta
    min_S_for_theta = 0.0
    outer_norms = np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                             math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    
    for i in range(N):
        V_rel = V_list[i] - np.array([cx, cy])
        proj = V_rel @ outer_norms.T
        max_proj = np.max(proj, axis=0)
        needed_apothem = np.max(max_proj)
        needed_S = needed_apothem / APOTHEM
        min_S_for_theta = max(min_S_for_theta, needed_S)
    
    # Try a few theta values to find the best S
    best_S = float('inf')
    best_theta = theta
    for t_candidate in [theta, theta + math.pi/6, theta + math.pi/12, theta - math.pi/12]:
        t_candidate = t_candidate % (math.pi / 3)
        outer_norms_c = np.array([[math.cos(t_candidate + math.pi / 6.0 + k * math.pi / 3.0),
                                   math.sin(t_candidate + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
        min_S_c = 0.0
        for i in range(N):
            V_rel = V_list[i] - np.array([cx, cy])
            proj = V_rel @ outer_norms_c.T
            max_proj = np.max(proj, axis=0)
            needed_apothem = np.max(max_proj)
            needed_S = needed_apothem / APOTHEM
            min_S_c = max(min_S_c, needed_S)
        if min_S_c < best_S:
            best_S = min_S_c
            best_theta = t_candidate
    
    # Check if the configuration is valid (no overlaps)
    is_valid = True
    for i in range(N):
        for j in range(i + 1, N):
            xi = inner_params[3 * i]
            yi = inner_params[3 * i + 1]
            ai = inner_params[3 * i + 2]
            xj = inner_params[3 * j]
            yj = inner_params[3 * j + 1]
            aj = inner_params[3 * j + 2]
            
            V_i = _hex_vertices(ai) + np.array([xi, yi])
            N_i = _hex_halfspace_normals(ai)
            V_j = _hex_vertices(aj) + np.array([xj, yj])
            N_j = _hex_halfspace_normals(aj)
            
            overlap = _pairwise_overlap_penalty(V_i, N_i, V_j, N_j)
            if overlap > 1e-6:
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid or best_S > 12:
        # Fall back to the initial lattice configuration
        inner = [[float(x), float(y), 0.0] for x, y, _ in initial_cartesian]
        return inner, [cx0, cy0], S0, 0.0
    
    inner = []
    for i in range(N):
        xi = float(inner_params[3 * i])
        yi = float(inner_params[3 * i + 1])
        ai_deg = float(math.degrees(inner_params[3 * i + 2] % (2 * math.pi)))
        inner.append([xi, yi, ai_deg])
    
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(math.degrees(best_theta))
    
    return inner, center, side, rotation
