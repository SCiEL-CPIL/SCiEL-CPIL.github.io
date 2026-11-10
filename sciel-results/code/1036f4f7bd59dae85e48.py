import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600 - 10  # Safety margin

    def hex_vertices(cx, cy, s, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(np.arange(0, 360, 60))
        return np.column_stack((
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ))

    def is_separated(v1, v2):
        # v1, v2 are (6,2) arrays
        def get_normals(verts):
            edges = np.roll(verts, -1, axis=0) - verts
            normals = np.column_stack([edges[:, 1], -edges[:, 0]])
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normals /= norms
            return normals
        
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        all_normals = np.vstack([n1, n2])
        
        for n in all_normals:
            p1 = v1 @ n
            p2 = v2 @ n
            if p1.max() < p2.min() - 1e-6 or p2.max() < p1.min() - 1e-6:
                return True
        return False

    def get_min_side_smooth(positions, angles, outer_angle=0.0, beta=50.0):
        # Smooth surrogate for min side using log-sum-exp
        a_rad = np.deg2rad(outer_angle)
        normal_angles = a_rad + np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Collect all projections
        all_projs = []
        for i in range(11):
            verts = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            projs = verts @ normals.T  # (6, 6)
            all_projs.append(projs)
        
        all_projs = np.concatenate(all_projs, axis=0)  # (66, 6)
        
        # For each normal direction k, find max projection across all vertices
        # Then we want max over k of max_proj_k
        max_per_normal = np.max(all_projs, axis=0)  # (6,)
        
        # Smooth max over the 6 normals
        max_proj = np.log(np.sum(np.exp(beta * max_per_normal))) / beta
        
        S = 2 * max_proj / np.sqrt(3)
        return S

    def get_min_side(positions, angles, outer_angle=0.0):
        a_rad = np.deg2rad(outer_angle)
        normal_angles = a_rad + np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        max_proj = 0
        for i in range(11):
            verts = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
            
        S = 2 * max_proj / np.sqrt(3)
        return S

    def objective(x):
        if time.time() > deadline:
            return 1e10
        
        positions = x[:22].reshape(11, 2)
        angles = x[22:33]
        
        S = get_min_side_smooth(positions, angles)
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                v1 = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
                v2 = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                if not is_separated(v1, v2):
                    d = np.linalg.norm(positions[i] - positions[j])
                    penalty += max(0, 2.0 - d) ** 2
                    
        return S + 100 * penalty

    # Initial guess: Honeycomb cluster
    sqrt3 = np.sqrt(3)
    init_pos = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        [3, sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [3, -sqrt3]
    ])
    init_angles = np.zeros(11)
    
    x0 = np.concatenate([init_pos.flatten(), init_angles])
    
    best_x = x0
    best_val = objective(x0)
    
    # Multiple restarts with perturbations
    rng = np.random.default_rng(42)
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Perturb the best solution
        noise_scale = 0.5 * (1 - r / n_restarts)
        x_perturbed = best_x + rng.normal(0, noise_scale, size=best_x.shape)
        
        try:
            result = minimize(objective, x_perturbed, method='Nelder-Mead', 
                              options={'maxiter': 15000, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_val:
                best_x = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 20:
        try:
            result = minimize(objective, best_x, method='Nelder-Mead', 
                              options={'maxiter': 20000, 'xatol': 1e-7, 'fatol': 1e-7})
            if result.fun < best_val:
                best_x = result.x
                best_val = result.fun
        except Exception:
            pass

    # Extract best result
    positions = best_x[:22].reshape(11, 2)
    angles = best_x[22:33]
    
    # Final validation with exact (non-smooth) side calculation
    S = get_min_side(positions, angles)
    
    valid = True
    if S < np.sqrt(11) - 1e-6:
        S = np.sqrt(11)
        
    # Check overlaps
    for i in range(11):
        for j in range(i+1, 11):
            v1 = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
            if not is_separated(v1, v2):
                valid = False
                break
        if not valid:
            break
            
    # Check containment
    if valid:
        outer_verts = hex_vertices(0, 0, S, 0.0)
        for i in range(11):
            v_inner = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                cross = dx * (v_inner[:, 1] - p1[1]) - dy * (v_inner[:, 0] - p1[0])
                if np.any(cross < -1e-6):
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # Fallback to seed
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
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = np.column_stack((positions, angles))
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = S

    return inner_hex_data, outer_hex_data, outer_hex_side_length
