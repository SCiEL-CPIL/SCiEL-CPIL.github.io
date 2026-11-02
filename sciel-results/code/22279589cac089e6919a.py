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

    # Generate a diverse set of initial configurations using force-directed layout
    # The idea: start with random positions, apply repulsive forces between hexagons
    # to push them apart, then compute the minimal enclosing hexagon.
    
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Define a set of candidate axial coordinate configurations
    candidates = []

    # Compact hexagonal clusters
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    # More diverse candidates
    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_28)

    # Compute the minimal enclosing hexagon side length for a given configuration
    def compute_enclosing_hexagon(pts, beta):
        """
        Compute the side length of the minimal enclosing regular hexagon
        with rotation beta for the given points.
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # For a given beta, the minimal enclosing hexagon center minimizes
        # the maximum absolute projection onto each normal direction.
        # This is a linear programming problem, but we can use the fact that
        # the optimal center is determined by the midpoint of the bounding
        # box in each normal direction.
        
        # First pass: estimate center as centroid
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        
        # Iterate to find optimal center
        for _ in range(50):
            max_proj = 0.0
            # Collect constraints: for each normal, the center must satisfy
            # |proj(center) - proj(pts)| <= D/2 for all pts
            # This means: min_proj <= proj(center) <= max_proj for each normal
            
            lo_x, lo_y = float('inf'), float('inf')
            hi_x, hi_y = float('-inf'), float('-inf')
            
            for nx, ny in normals:
                projs = pts[:, 0] * nx + pts[:, 1] * ny
                p_min = np.min(projs)
                p_max = np.max(projs)
                # center projection must be in [p_min, p_max]
                # cx*nx + cy*ny >= p_min and cx*nx + cy*ny <= p_max
                
            # Simple approach: use the intersection of half-planes
            # For now, use iterative projection
            for nx, ny in normals:
                projs = pts[:, 0] * nx + pts[:, 1] * ny
                p_min = np.min(projs)
                p_max = np.max(projs)
                center_proj = cx * nx + cy * ny
                # If center projection is outside [p_min, p_max], move it
                if center_proj < p_min:
                    # Move center in direction of normal to increase projection
                    cx += (p_min - center_proj) * nx
                    cy += (p_min - center_proj) * ny
                elif center_proj > p_max:
                    cx -= (center_proj - p_max) * nx
                    cy -= (center_proj - p_max) * ny
            
            # Compute max_proj with new center
            max_proj = 0.0
            for nx, ny in normals:
                projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
            D = max_proj
            S = (D + APOTHEM) / APOTHEM
        
        return S, (cx, cy), D

    def optimize_shape(pts):
        """
        Optimize the enclosing hexagon for a given set of points.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = compute_enclosing_hexagon(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: compute_enclosing_hexagon(pts, b)[0], 
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = compute_enclosing_hexagon(pts, best_beta)
        except Exception:
            pass
        
        # Local optimization of center and beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0.0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = abs((px - cx) * nx + (py - cy) * ny)
                            if proj > max_proj:
                                max_proj = proj
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    # Force-directed layout to generate new configurations
    def force_directed_layout(initial_pts, iterations=200):
        """
        Apply force-directed layout to push hexagons apart.
        """
        pts = initial_pts.copy()
        n = len(pts)
        
        for it in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            forces = np.zeros_like(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 2.0:  # Only apply force if close
                        force_mag = 1.0 / (dist + 0.01)
                        forces[i] += force_mag * diff / (dist + 1e-10)
                        forces[j] -= force_mag * diff / (dist + 1e-10)
            
            # Add a mild centering force to keep the cluster compact
            centroid = np.mean(pts, axis=0)
            for i in range(n):
                forces[i] -= 0.01 * (pts[i] - centroid)
            
            pts += 0.05 * forces
        
        return pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        
        # Try both the original layout and a force-directed version
        for trial_pts in [pts, force_directed_layout(pts, 100)]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, beta, center = optimize_shape(trial_pts)
            if S < best_global_S:
                best_global_S = S
                best_global_pts = trial_pts.copy()
                best_global_beta = beta
                best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    # Additional optimization: try to improve the best configuration found
    if best_global_pts is not None and best_global_center is not None:
        # Try rotating individual hexagons and see if we can improve
        # For now, just do a final polish on the best solution
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pts_final = best_global_pts.copy()
            cx, cy = best_global_center
            beta_final = best_global_beta
            
            # Final local optimization
            def final_objective(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                normals = []
                for k in range(6):
                    ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                max_proj = 0.0
                for nx, ny in normals:
                    for px, py in pts_final:
                        proj = abs((px - cx) * nx + (py - cy) * ny)
                        if proj > max_proj:
                            max_proj = proj
                
                S = (max_proj + APOTHEM) / APOTHEM
                return S
            
            init = [cx, cy, beta_final]
            try:
                res = minimize(final_objective, init, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < best_global_S:
                    best_global_S = res.fun
                    best_global_center = (res.x[0], res.x[1])
                    best_global_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Use a very small safety margin to ensure no overlap
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recompute the enclosing hexagon with the scaled points
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
