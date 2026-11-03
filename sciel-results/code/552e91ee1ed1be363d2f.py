import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial to Cartesian conversion
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Compute minimal enclosing hexagon side length for a set of points with given beta and center
    def calc_S(pts, beta, cx, cy):
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

    # Force-directed relaxation to find good local configurations
    def relax(pts, n_iter=2000, temperature=0.5):
        pts = pts.copy()
        best_S = float('inf')
        best_pts = pts.copy()
        best_beta = 0.0
        best_center = None
        
        for it in range(n_iter):
            # Compute forces
            forces = np.zeros_like(pts)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 2.0:
                        # Repulsive force
                        f = (2.0 - dist) * 10.0
                        fx = f * dx / (dist + 1e-9)
                        fy = f * dy / (dist + 1e-9)
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            # Update positions
            for i in range(len(pts)):
                pts[i, 0] += forces[i, 0] * temperature
                pts[i, 1] += forces[i, 1] * temperature
            temperature *= 0.999
            
            # Center the configuration
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            pts[:, 0] -= cx
            pts[:, 1] -= cy
            
            # Compute S for current configuration
            if it % 10 == 0:
                S = calc_S(pts, 0.0, 0.0, 0.0)
                if S < best_S:
                    best_S = S
                    best_pts = pts.copy()
                    best_center = (0.0, 0.0)
        
        return best_S, best_pts, best_center

    # Optimize beta and center for a fixed point set
    def optimize_beta_center(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        
        # Grid search over beta
        for i in range(72):
            beta = i * (math.pi / 6) / 71
            # For each beta, find optimal center by minimizing max projection
            # Use simple iterative method
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            
            for _ in range(50):
                normals = []
                for k in range(6):
                    ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                # Find min and max projections for each normal
                min_projs = []
                max_projs = []
                for nx, ny in normals:
                    projs = pts[:, 0] * nx + pts[:, 1] * ny
                    min_projs.append(np.min(projs))
                    max_projs.append(np.max(projs))
                
                # Optimal center is midpoint of min and max for each direction
                cx_new = 0.0
                cy_new = 0.0
                count = 0
                for k in range(6):
                    nx, ny = normals[k]
                    cx_new += (max_projs[k] + min_projs[k]) / (2.0 * nx)
                    cy_new += (max_projs[k] + min_projs[k]) / (2.0 * ny)
                    count += 1
                cx_new /= count
                cy_new /= count
                
                # Update
                cx = 0.5 * cx + 0.5 * cx_new
                cy = 0.5 * cy + 0.5 * cy_new
            
            S = calc_S(pts, beta, cx, cy)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = (cx, cy)
        
        # Refine with local optimization
        try:
            def objective(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                return calc_S(pts, beta, cx, cy)
            
            starts = [
                [best_center[0], best_center[1], best_beta],
                [best_center[0] + 0.1, best_center[1], best_beta],
                [best_center[0] - 0.1, best_center[1], best_beta],
                [best_center[0], best_center[1] + 0.1, best_beta],
                [best_center[0], best_center[1] - 0.1, best_beta],
                [best_center[0], best_center[1], best_beta + 0.05],
                [best_center[0], best_center[1], best_beta - 0.05],
            ]
            
            for init in starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                res = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    # Generate candidate configurations
    def generate_candidates():
        candidates = []
        
        # Hexagonal lattice patterns
        # Pattern 1: Center + 6 around + 4 more
        axial_1 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        candidates.append(axial_1)
        
        # Pattern 2: 3x3 grid
        axial_2 = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (-1, 1), (3, 1)
        ]
        candidates.append(axial_2)
        
        # Pattern 3: 4 in a row + 3 above + 4 below
        axial_3 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1),
            (1, 2), (2, 2), (3, 2),
        ]
        candidates.append(axial_3)
        
        # Pattern 4: Compact cluster
        axial_4 = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3), (2, 3)
        ]
        candidates.append(axial_4)
        
        # Pattern 5: Star shape
        axial_5 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 1), (-2, 2), (-1, 3), (1, -2)
        ]
        candidates.append(axial_5)
        
        # Pattern 6: Diamond
        axial_6 = [
            (0, 0), (1, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3), (2, 3)
        ]
        candidates.append(axial_6)
        
        # Pattern 7: L-shape
        axial_7 = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2),
            (0, 3), (1, 3),
            (0, 4)
        ]
        candidates.append(axial_7)
        
        # Pattern 8: T-shape
        axial_8 = [
            (0, 0), (1, 0), (2, 0),
            (1, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3),
            (0, 4), (1, 4), (2, 4)
        ]
        candidates.append(axial_8)
        
        # Pattern 9: Cross
        axial_9 = [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (1, 1), (-1, -1)
        ]
        candidates.append(axial_9)
        
        # Pattern 10: Zigzag
        axial_10 = [
            (0, 0), (1, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2),
            (0, 3), (1, 3), (2, 3),
            (1, 4)
        ]
        candidates.append(axial_10)
        
        return candidates

    def axial_list_to_pts(axial_list):
        pts = []
        for q, r in axial_list:
            x, y = axial_to_cartesian(q, r)
            pts.append([x, y])
        return np.array(pts)

    # Main optimization loop
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    candidates = generate_candidates()
    
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = axial_list_to_pts(axial)
        
        # Relax the configuration
        S_relaxed, pts_relaxed, _ = relax(pts, n_iter=1000, temperature=0.3)
        
        # Optimize beta and center
        S, beta, center = optimize_beta_center(pts_relaxed)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_relaxed
            best_global_beta = beta
            best_global_center = center

    # Fallback if no good configuration found
    if best_global_center is None or best_global_S == float('inf'):
        pts = axial_list_to_pts(candidates[0])
        best_global_pts = pts
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = calc_S(pts, best_global_beta, best_global_center[0], best_global_center[1])

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Scale points slightly to ensure no overlap
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    # Recompute S with scaled points
    max_proj = 0.0
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
