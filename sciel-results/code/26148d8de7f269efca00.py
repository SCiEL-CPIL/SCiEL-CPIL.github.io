import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial import ConvexHull

def hexagon_packing_11():
    start_time = time.time()
    deadline = 625.0
    
    # Unit hexagon vertices (angle 0)
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # Edge normals at 30, 90, 150, 210, 270, 330
    edge_normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(30, 360, 60)
    ])
    
    # Projections of unit hexagon vertices onto edge normals
    proj_hex = hex_verts @ edge_normals.T
    min_hex = np.min(proj_hex, axis=0)
    max_hex = np.max(proj_hex, axis=0)
    
    def get_vertices(center, angle_deg=0):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + hex_verts @ rot.T

    def max_penetration(c1, c2):
        c_diff = c1 - c2
        proj_c = c_diff @ edge_normals.T
        min2_shifted = min_hex + proj_c
        max2_shifted = max_hex + proj_c
        overlaps = np.minimum(max_hex, max2_shifted) - np.maximum(min_hex, min2_shifted)
        return np.max(overlaps)

    def compute_side(centers, theta):
        outer_normals = np.array([[np.cos(theta + np.deg2rad(60 * i)), 
                                   np.sin(theta + np.deg2rad(60 * i))] for i in range(6)])
        proj_hex_outer = hex_verts @ outer_normals.T
        max_proj_hex = np.max(proj_hex_outer, axis=0)
        proj_centers = centers @ outer_normals.T
        max_proj_centers = np.max(proj_centers, axis=0)
        h_total = max_proj_centers + max_proj_hex
        max_h = np.max(h_total)
        side = (2.0 / np.sqrt(3.0)) * max_h
        return side

    def is_valid(centers, theta):
        # Check all pairs are separated
        for i in range(11):
            for j in range(i + 1, 11):
                if max_penetration(centers[i], centers[j]) > 1e-6:
                    return False
        return True

    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        centers = params[:22].reshape(11, 2)
        theta = params[22]
        side = compute_side(centers, theta)
        return side

    # Constraint: no overlaps between any pair
    def separation_constraints(params):
        centers = params[:22].reshape(11, 2)
        constraints = []
        for i in range(11):
            for j in range(i + 1, 11):
                pen = max_penetration(centers[i], centers[j])
                # We need pen <= 0 (with margin), so constraint is -pen >= 0
                constraints.append(-pen - 1e-7)
        return np.array(constraints)

    # Known good initial configuration: center + 6 ring1 + 4 ring2
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
    
    # Try different symmetric subsets of ring2
    idx_sets = [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 3, 6, 9],
        [0, 2, 6, 8],
        [0, 5, 6, 11],
        [1, 4, 7, 10],
        [2, 4, 6, 8],
        [3, 5, 7, 9],
    ]
    
    best_result = None
    best_score = float('inf')
    
    for idxs in idx_sets:
        if time.time() > deadline - 30:
            break
            
        r2_pts = ring2[idxs]
        centers = np.vstack([base_centers, r2_pts])
        init_params = np.concatenate([centers.flatten(), [0.0]])
        
        # First, check if this initial config is valid
        if is_valid(centers, 0.0):
            side = compute_side(centers, 0.0)
            if side < best_score:
                best_score = side
                inner_hex_data = np.column_stack([centers, np.zeros(11)])
                outer_hex_data = np.array([0, 0, 0.0])
                best_result = (inner_hex_data, outer_hex_data, best_score)
        
        # Optimize with SLSQP
        try:
            res = minimize(objective, init_params, method='SLSQP',
                           constraints=[{'type': 'ineq', 'fun': separation_constraints}],
                           options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.fun < best_score:
                centers = res.x[:22].reshape(11, 2)
                theta = res.x[22]
                
                if is_valid(centers, theta):
                    side = compute_side(centers, theta)
                    if side < best_score:
                        best_score = side
                        inner_hex_data = np.column_stack([centers, np.zeros(11)])
                        outer_hex_data = np.array([0, 0, np.rad2deg(theta)])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        
        except Exception:
            pass
    
    # Second phase: try Nelder-Mead from best result for further refinement
    if best_result is not None and time.time() < deadline - 60:
        base_centers = best_result[0][:, :2]
        base_theta = np.deg2rad(best_result[1][2])
        
        # Local search with small perturbations
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            
            scale = 0.005
            perturbed_centers = base_centers + np.random.randn(11, 2) * scale
            perturbed_theta = base_theta + np.random.randn() * scale * 0.1
            
            init_params = np.concatenate([perturbed_centers.flatten(), [perturbed_theta]])
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
                
                centers = res.x[:22].reshape(11, 2)
                theta = res.x[22]
                
                if is_valid(centers, theta):
                    side = compute_side(centers, theta)
                    if side < best_score:
                        best_score = side
                        inner_hex_data = np.column_stack([centers, np.zeros(11)])
                        outer_hex_data = np.array([0, 0, np.rad2deg(theta)])
                        best_result = (inner_hex_data, outer_hex_data, best_score)
                        base_centers = centers
                        base_theta = theta
                        
            except Exception:
                pass
    
    if best_result is None:
        # Fallback
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
