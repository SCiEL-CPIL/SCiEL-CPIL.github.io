import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin from 150s kill
    sqrt3 = math.sqrt(3)

    # Precompute hexagon vertices relative to center (unit side)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    # Generate candidate centers for inner hexagons
    # A natural packing: center + ring 1 (6) + ring 2 (12)
    d = sqrt3  # distance between adjacent hexagon centers
    
    # Center
    centers = [(0.0, 0.0)]
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
    
    # Ring 2 (12 hexagons) - using axial coordinates
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (sqrt3 / 2.0) * r
                ring2_centers.append((x, y))
    
    # We need 11 hexagons. Start with 7 (center + ring 1), pick 4 from ring 2
    base_centers = centers[:7]
    
    # Generate all combinations of 4 from ring 2
    import itertools
    all_combinations = list(itertools.combinations(range(12), 4))
    
    # For each combination, we'll try to optimize positions and rotations
    # But first, let's try a simpler approach: fix the 7 base centers, optimize the 4 ring-2 centers
    
    def compute_outer_side(centers, angles, outer_angle=0.0):
        """Compute the minimum side length of outer hexagon containing all inner hexagons."""
        # Outer hexagon has apothem A, side = 2A/sqrt(3)
        # For each inner hexagon, compute max distance from origin in direction of each outer normal
        # Outer normals at angles outer_angle + k*60°
        
        max_proj = 0.0
        for i in range(len(centers)):
            cx, cy = centers[i]
            theta = angles[i]
            # Rotate vertices
            c, s = math.cos(theta), math.sin(theta)
            rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
            ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
            px = rx + cx
            py = ry + cy
            # Project onto 6 outer normals
            for k in range(6):
                ang = outer_angle + k * math.pi / 3
                nx = math.cos(ang)
                ny = math.sin(ang)
                proj = np.max(np.abs(px * nx + py * ny))
                if proj > max_proj:
                    max_proj = proj
        
        # Apothem = max_proj, side = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def check_overlap(centers, angles):
        """Check if any two hexagons overlap using SAT."""
        def get_hex_verts(cx, cy, angle):
            c, s = math.cos(angle), math.sin(angle)
            rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
            ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
            return np.stack([rx + cx, ry + cy], axis=-1)
        
        def sat_overlap(v1, v2):
            for v in (v1, v2):
                for i in range(6):
                    p1 = v[i]
                    p2 = v[(i + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                        return False
            return True
        
        verts_list = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(len(centers))]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if sat_overlap(verts_list[i], verts_list[j]):
                    return True
        return False

    def overlap_penalty(centers, angles):
        """Compute penalty for overlaps."""
        penalty = 0.0
        min_dist = sqrt3  # minimum center distance for non-overlapping unit hexagons
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        return penalty

    best_S = float('inf')
    best_centers = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Strategy: For each combination of 4 ring-2 centers, optimize positions and rotations
    # Use Nelder-Mead for local optimization
    
    np.random.seed(42)
    
    for combo in all_combinations:
        if time.time() > deadline - 10:
            break
        
        # Initial centers: 7 base + 4 selected ring-2
        init_centers = list(base_centers)
        for idx in combo:
            init_centers.append(ring2_centers[idx])
        
        # Initial angles: all 0
        init_angles = [0.0] * 11
        
        # Optimize: parameters are [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, outer_angle]
        # Total: 11*3 + 1 = 34 parameters
        
        def objective(params):
            if time.time() > deadline - 5:
                return 1e10
            
            xs = params[0::3]
            ys = params[1::3]
            thetas = params[2::3]
            outer_angle = params[-1]
            
            centers_opt = list(zip(xs, ys))
            
            S = compute_outer_side(centers_opt, thetas, outer_angle)
            pen = overlap_penalty(centers_opt, thetas)
            
            return S + 100.0 * pen
        
        init_params = []
        for cx, cy in init_centers:
            init_params.extend([cx, cy, 0.0])
        init_params.append(0.0)  # outer angle
        init_params = np.array(init_params)
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_S:
                final_params = res.x
                xs = final_params[0::3]
                ys = final_params[1::3]
                thetas = final_params[2::3]
                outer_angle = final_params[-1]
                
                centers_final = list(zip(xs, ys))
                pen = overlap_penalty(centers_final, thetas)
                
                if pen < 1e-3:  # Valid packing
                    S_final = compute_outer_side(centers_final, thetas, outer_angle)
                    if S_final < best_S:
                        best_S = S_final
                        best_centers = centers_final
                        best_angles = thetas
                        best_outer_angle = outer_angle
                        
        except Exception:
            continue
    
    # If no valid packing found, try a different strategy: optimize all 11 freely
    if best_S == float('inf') and time.time() < deadline - 15:
        # Start with a more scattered initial guess
        init_centers = []
        for i in range(11):
            if i < 7:
                init_centers.append(base_centers[i])
            else:
                # Place remaining 4 in a larger ring
                angle = (i - 6) * math.pi / 3
                x = 2 * d * math.cos(angle)
                y = 2 * d * math.sin(angle)
                init_centers.append((x, y))
        
        def objective2(params):
            if time.time() > deadline - 5:
                return 1e10
            
            xs = params[0::3]
            ys = params[1::3]
            thetas = params[2::3]
            outer_angle = params[-1]
            
            centers_opt = list(zip(xs, ys))
            
            S = compute_outer_side(centers_opt, thetas, outer_angle)
            pen = overlap_penalty(centers_opt, thetas)
            
            return S + 100.0 * pen
        
        init_params = []
        for cx, cy in init_centers:
            init_params.extend([cx, cy, 0.0])
        init_params.append(0.0)
        init_params = np.array(init_params)
        
        try:
            res = minimize(objective2, init_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_S:
                final_params = res.x
                xs = final_params[0::3]
                ys = final_params[1::3]
                thetas = final_params[2::3]
                outer_angle = final_params[-1]
                
                centers_final = list(zip(xs, ys))
                pen = overlap_penalty(centers_final, thetas)
                
                if pen < 1e-3:
                    S_final = compute_outer_side(centers_final, thetas, outer_angle)
                    if S_final < best_S:
                        best_S = S_final
                        best_centers = centers_final
                        best_angles = thetas
                        best_outer_angle = outer_angle
                        
        except Exception:
            pass
    
    # If still no valid packing, return a fallback
    if best_S == float('inf') or best_centers is None:
        # Fallback: place hexagons in a line (not optimal but valid)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Construct output
    inner = []
    for i in range(11):
        x, y = best_centers[i]
        ang = math.degrees(best_angles[i])
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(best_S), float(math.degrees(best_outer_angle))
