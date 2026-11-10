import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    
    # Unit hexagon vertices (angle 0)
    # Vertices at 0, 60, ... 300 degrees
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Precompute normals for overlap checking
    # Edge normals are at 30, 90, 150, 210, 270, 330
    edge_normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(30, 360, 60)
    ])
    
    # Projections of unit hexagon vertices onto edge normals
    # This is constant for all unit hexagons
    proj_hex = hex_verts @ edge_normals.T
    min_hex = np.min(proj_hex, axis=0)
    max_hex = np.max(proj_hex, axis=0)
    
    def get_vertices(center, angle_deg=0):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + hex_verts @ rot.T

    def max_penetration(c1, c2):
        # Returns the maximum penetration depth between two unit hexagons at c1 and c2
        # They are aligned.
        c_diff = c1 - c2
        proj_c = c_diff @ edge_normals.T
        
        # Shifted intervals for hex2
        min2_shifted = min_hex + proj_c
        max2_shifted = max_hex + proj_c
        
        # Overlap in each normal direction
        overlaps = np.minimum(max_hex, max2_shifted) - np.maximum(min_hex, min2_shifted)
        
        return np.max(overlaps)

    def total_penalty(centers):
        # Sum of squared penetration depths for all pairs
        pen_sum = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                pen = max_penetration(centers[i], centers[j])
                if pen > 0:
                    pen_sum += pen * pen
        return pen_sum

    def objective(params):
        # params: [x1, y1, ..., x11, y11, theta]
        # theta is in radians
        if time.time() > deadline - 5:
            return 1e10
            
        centers = params[:22].reshape(11, 2)
        theta = params[22]
        
        # Calculate side length
        # Normals for outer hexagon
        outer_normals = []
        for i in range(6):
            a = theta + np.deg2rad(60 * i)
            outer_normals.append([np.cos(a), np.sin(a)])
        outer_normals = np.array(outer_normals)
        
        # Support function of the union of hexagons
        # h_total(n) = max_i (n . c_i + h_hex(n))
        
        # h_hex(n) for unit hexagon
        proj_hex_outer = hex_verts @ outer_normals.T
        max_proj_hex = np.max(proj_hex_outer, axis=0)
        
        proj_centers = centers @ outer_normals.T
        max_proj_centers = np.max(proj_centers, axis=0)
        
        h_total = max_proj_centers + max_proj_hex
        max_h = np.max(h_total)
        
        side = (2.0 / np.sqrt(3.0)) * max_h
        
        # Penalty
        pen = total_penalty(centers)
        
        # Large penalty weight to ensure non-overlap
        weight = 100.0
        return side + weight * pen

    # Initial candidates
    # 1. Center + Ring 1 (7 hexagons) + 4 Ring 2
    d = np.sqrt(3)
    v = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        v.append([d * np.cos(angle), d * np.sin(angle)])
    v = np.array(v)
    
    ring2 = []
    for i in range(6):
        ring2.append(2 * v[i])
        ring2.append(v[i] + v[(i + 1) % 6])
    ring2 = np.array(ring2)
    
    base_centers = np.vstack([np.array([[0, 0]]), v])
    
    candidates = []
    
    # Symmetric choices for 4 ring 2 points:
    idx_sets = [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 3, 6, 9],
        [0, 2, 6, 8],
        [0, 5, 6, 11],
        [1, 4, 7, 10]
    ]
    
    for idxs in idx_sets:
        r2_pts = ring2[idxs]
        centers = np.vstack([base_centers, r2_pts])
        # Flatten centers
        init_params = np.concatenate([centers.flatten(), [0.0]])
        candidates.append(init_params)
    
    # Add some randomized initial candidates
    np.random.seed(42)
    for _ in range(20):
        centers = np.random.randn(11, 2) * 2.0
        # Ensure some structure
        centers[0] = [0, 0]
        for i in range(1, 7):
            angle = np.deg2rad(60 * (i - 1))
            centers[i] = [d * np.cos(angle), d * np.sin(angle)]
        # Randomize the last 4
        for i in range(7, 11):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(1.5, 3.5)
            centers[i] = [r * np.cos(angle), r * np.sin(angle)]
        
        init_params = np.concatenate([centers.flatten(), [np.random.uniform(-np.pi/6, np.pi/6)]])
        candidates.append(init_params)
    
    best_result = None
    best_score = float('inf')
    
    # First pass: Nelder-Mead
    for init in candidates:
        if time.time() > deadline - 60:
            break
            
        try:
            # Use Nelder-Mead
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'disp': False})
            
            if res.success or res.fun < best_score:
                # Check validity
                centers = res.x[:22].reshape(11, 2)
                theta = res.x[22]
                
                # Verify no overlaps
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        if max_penetration(centers[i], centers[j]) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Calculate final side
                    outer_normals = []
                    for i in range(6):
                        a = theta + np.deg2rad(60 * i)
                        outer_normals.append([np.cos(a), np.sin(a)])
                    outer_normals = np.array(outer_normals)
                    
                    proj_hex = hex_verts @ outer_normals.T
                    max_proj_hex = np.max(proj_hex, axis=0)
                    
                    proj_centers = centers @ outer_normals.T
                    max_proj_centers = np.max(proj_centers, axis=0)
                    
                    h_total = max_proj_centers + max_proj_hex
                    max_h = np.max(h_total)
                    side = (2.0 / np.sqrt(3.0)) * max_h
                    
                    if side < best_score:
                        best_score = side
                        inner_hex_data = np.column_stack([centers, np.zeros(11)])
                        outer_hex_data = np.array([0, 0, np.rad2deg(theta)])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        
        except Exception as e:
            pass

    # Second pass: Try L-BFGS-B with better initial points from first pass
    if best_result is not None and time.time() < deadline - 120:
        # Use the best result as starting point for L-BFGS-B
        init_params = np.concatenate([best_result[0][:, :2].flatten(), 
                                       [np.deg2rad(best_result[1][2])]])
        
        try:
            res = minimize(objective, init_params, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                centers = res.x[:22].reshape(11, 2)
                theta = res.x[22]
                
                # Verify no overlaps
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        if max_penetration(centers[i], centers[j]) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    outer_normals = []
                    for i in range(6):
                        a = theta + np.deg2rad(60 * i)
                        outer_normals.append([np.cos(a), np.sin(a)])
                    outer_normals = np.array(outer_normals)
                    
                    proj_hex = hex_verts @ outer_normals.T
                    max_proj_hex = np.max(proj_hex, axis=0)
                    
                    proj_centers = centers @ outer_normals.T
                    max_proj_centers = np.max(proj_centers, axis=0)
                    
                    h_total = max_proj_centers + max_proj_hex
                    max_h = np.max(h_total)
                    side = (2.0 / np.sqrt(3.0)) * max_h
                    
                    if side < best_score:
                        best_score = side
                        inner_hex_data = np.column_stack([centers, np.zeros(11)])
                        outer_hex_data = np.array([0, 0, np.rad2deg(theta)])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        
        except Exception as e:
            pass

    # Third pass: Local search around best result
    if best_result is not None and time.time() < deadline - 60:
        base_centers = best_result[0][:, :2]
        base_theta = np.deg2rad(best_result[1][2])
        
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random perturbation
            scale = 0.01
            perturbed_centers = base_centers + np.random.randn(11, 2) * scale
            perturbed_theta = base_theta + np.random.randn() * scale * 0.1
            
            init_params = np.concatenate([perturbed_centers.flatten(), [perturbed_theta]])
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res.fun < best_score:
                    centers = res.x[:22].reshape(11, 2)
                    theta = res.x[22]
                    
                    # Verify no overlaps
                    valid = True
                    for i in range(11):
                        for j in range(i + 1, 11):
                            if max_penetration(centers[i], centers[j]) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        outer_normals = []
                        for i in range(6):
                            a = theta + np.deg2rad(60 * i)
                            outer_normals.append([np.cos(a), np.sin(a)])
                        outer_normals = np.array(outer_normals)
                        
                        proj_hex = hex_verts @ outer_normals.T
                        max_proj_hex = np.max(proj_hex, axis=0)
                        
                        proj_centers = centers @ outer_normals.T
                        max_proj_centers = np.max(proj_centers, axis=0)
                        
                        h_total = max_proj_centers + max_proj_hex
                        max_h = np.max(h_total)
                        side = (2.0 / np.sqrt(3.0)) * max_h
                        
                        if side < best_score:
                            best_score = side
                            inner_hex_data = np.column_stack([centers, np.zeros(11)])
                            outer_hex_data = np.array([0, 0, np.rad2deg(theta)])
                            best_result = (inner_hex_data, outer_hex_data, best_score)
                            base_centers = centers
                            base_theta = theta
                            
            except Exception as e:
                pass

    if best_result is None:
        # Fallback to a known valid configuration (Seed-like)
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result
