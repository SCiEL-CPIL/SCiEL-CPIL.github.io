import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define a wide variety of candidate configurations (axial coordinates)
    # Each configuration is a list of (q, r) pairs.
    # We assume all inner hexagons have the same orientation (0 degrees relative to each other).
    # The outer hexagon rotates.
    
    candidates = []
    
    # 1. Compact flower (current best candidate)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ])
    
    # 2. 3x3 block minus 2 corners
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ])
    
    # 3. Rectangular 3x4 minus 1
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 4. Linear + 2
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (-1, 0), (-2, 0), (-3, 0), (-4, 0),
        (0, 1), (0, -1)
    ])
    
    # 5. Skewed 3x3
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ])
    
    # 6. Center + 6 + 4 in a line
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (3, 0), (4, 0), (5, 0)
    ])
    
    # 7. 4x3 block
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 8. T-shape
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1),
        (1, 2), (2, 2),
        (1, -1), (2, -1)
    ])
    
    # 9. Cross
    candidates.append([
        (0, 0),
        (1, 0), (-1, 0), (2, 0), (-2, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, -1),
        (1, -1), (-1, 1)
    ])
    
    # 10. 2x5 rectangle
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)
    ])
    
    # 11. 3x3 square (hex lattice)
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3) # Wait, this is 11 points. Let's verify coords.
    ])
    # Actually, let's just generate a few more random-ish but valid lattice sets
    # We need 11 points.
    
    # 12. Long bar with bumps
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (1, 1), (2, 1),
        (1, -1), (2, -1),
        (3, 1)
    ])
    
    # 13. Compact triangle
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3)
    ])
    
    # 14. Another flower variant
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ])
    
    # 15. 5x2 block
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1)
    ])
    
    # 16. 3x4 block minus corner
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])

    best_result = None
    best_S = float('inf')
    
    # Helper to convert axial to cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
    
    # Helper to solve LP for given beta and points
    def solve_lp(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
            
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
            
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # Constraint: n . (p + c) <= D - a  =>  -n . c - D <= -n . p + a
                # Wait, standard form for linprog is A_ub x <= b_ub.
                # Variables: x = [cx, cy, D]
                # Constraint: n . (p_i + c) <= D - a
                # n . p_i + n . c <= D - a
                # -n . c - D <= -n . p_i - a
                # Also: -n . (p_i + c) <= D - a
                # -n . p_i - n . c <= D - a
                # n . c + D >= -n . p_i + a  =>  -n . c - D <= n . p_i - a
                
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny) - APOTHEM)
                
                A_ub.append([nx, ny, -1.0])
                b_ub.append((px * nx + py * ny) - APOTHEM)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    for config_idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        
        # Coarse search for beta
        best_beta_local = 0.0
        best_S_local = float('inf')
        best_center_local = None
        
        num_steps = 36
        for i in range(num_steps + 1):
            beta = i * (math.pi / 6.0) / num_steps
            S, center, D = solve_lp(pts, beta)
            if S < best_S_local:
                best_S_local = S
                best_beta_local = beta
                best_center_local = center
                
        if best_S_local == float('inf'):
            continue
            
        # Refine beta using minimize_scalar
        # Define objective function
        def obj(beta):
            S, _, _ = solve_lp(pts, beta)
            return S
            
        try:
            # Window around best coarse beta
            # Step size was (pi/6)/36 approx 0.0145
            # Use a window of +/- 2 steps
            step = (math.pi / 6.0) / num_steps
            lower = max(0.0, best_beta_local - 2 * step)
            upper = min(math.pi / 6.0, best_beta_local + 2 * step)
            
            res = minimize_scalar(obj, bounds=(lower, upper), method='bounded',
                                  options={'maxiter': 100, 'xatol': 1e-8})
            if res.fun < best_S_local:
                best_S_local = res.fun
                best_beta_local = res.x
                # Get center for this best beta
                _, best_center_local, _ = solve_lp(pts, best_beta_local)
        except Exception:
            pass
            
        # Final polish with tighter tolerance
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                lower = max(0.0, best_beta_local - 0.005)
                upper = min(math.pi / 6.0, best_beta_local + 0.005)
                res2 = minimize_scalar(obj, bounds=(lower, upper), method='bounded',
                                       options={'maxiter': 200, 'xatol': 1e-10})
                if res2.fun < best_S_local:
                    best_S_local = res2.fun
                    best_beta_local = res2.x
                    _, best_center_local, _ = solve_lp(pts, best_beta_local)
            except Exception:
                pass

        if best_S_local < best_S:
            best_S = best_S_local
            best_result = (best_center_local, best_beta_local, pts.copy())
            
    # Fallback
    if best_result is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_cx, best_cy, best_beta, best_pts = best_result
    
    # Prepare output
    inner = []
    angle_deg = math.degrees(best_beta)
    for x, y in best_pts:
        inner.append([float(x), float(y), angle_deg])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = angle_deg
    
    return inner, center, side, rotation
