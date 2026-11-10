import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds safety margin
    
    # Constants
    n_inner = 11
    sqrt3 = np.sqrt(3)
    
    # Helper functions
    
    def get_vertices_vectorized(params):
        """
        params: (N, 3) array of (x, y, angle_deg)
        Returns: (N, 6, 2) array of vertices
        """
        cx = params[:, 0]
        cy = params[:, 1]
        angles = np.deg2rad(params[:, 2])
        
        # Precompute cos/sin for 6 vertices
        base_angles = np.arange(6) * np.pi / 3
        all_angles = angles[:, np.newaxis] + base_angles[np.newaxis, :]
        
        cos_vals = np.cos(all_angles)
        sin_vals = np.sin(all_angles)
        
        cx_col = cx[:, np.newaxis]
        cy_col = cy[:, np.newaxis]
        
        x_verts = cx_col + cos_vals
        y_verts = cy_col + sin_vals
        
        vertices = np.stack((x_verts, y_verts), axis=-1)
        return vertices

    def compute_outer_side(vertices, Cx, Cy, alpha_deg):
        """
        vertices: (N, 6, 2)
        Cx, Cy: center of outer hex
        alpha_deg: angle of outer hex
        Returns: minimal side length S
        """
        alpha_rad = np.deg2rad(alpha_deg)
        normal_angles = alpha_rad + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.stack((np.cos(normal_angles), np.sin(normal_angles)), axis=-1)
        
        shifted = vertices - np.array([Cx, Cy])
        flat_verts = shifted.reshape(-1, 2)
        
        projections = flat_verts @ normals.T
        max_proj = np.max(projections, axis=0)
        
        S = 2 * np.max(max_proj) / sqrt3
        return S

    def check_overlaps(params):
        """
        params: (N, 3)
        Returns: True if no overlaps, False otherwise.
        """
        vertices = get_vertices_vectorized(params)
        N = params.shape[0]
        
        angles = np.deg2rad(params[:, 2])
        
        for i in range(N):
            for j in range(i + 1, N):
                v_i = vertices[i]
                v_j = vertices[j]
                
                ang_i = angles[i]
                norm_ang_i = ang_i + np.pi/6 + np.arange(6) * np.pi/3
                norms_i = np.stack((np.cos(norm_ang_i), np.sin(norm_ang_i)), axis=-1)
                
                ang_j = angles[j]
                norm_ang_j = ang_j + np.pi/6 + np.arange(6) * np.pi/3
                norms_j = np.stack((np.cos(norm_ang_j), np.sin(norm_ang_j)), axis=-1)
                
                all_norms = np.vstack((norms_i, norms_j))
                
                proj_i = v_i @ all_norms.T
                proj_j = v_j @ all_norms.T
                
                min_i = np.min(proj_i, axis=0)
                max_i = np.max(proj_i, axis=0)
                min_j = np.min(proj_j, axis=0)
                max_j = np.max(proj_j, axis=0)
                
                gaps = np.minimum(min_j - max_i, min_i - max_j)
                
                if np.max(gaps) > 1e-6:
                    continue
                
                return False
                
        return True

    def objective(params):
        inner_params = params[:33].reshape(11, 3)
        Cx = params[33]
        Cy = params[34]
        alpha = params[35]
        
        valid = check_overlaps(inner_params)
        
        vertices = get_vertices_vectorized(inner_params)
        S = compute_outer_side(vertices, Cx, Cy, alpha)
        
        if not valid:
            return S + 1000.0
            
        return S

    # Initial guess
    init_inner = np.array([
        [0, 0, 0],
        [sqrt3, 0, 0],
        [sqrt3/2, 1.5, 0],
        [-sqrt3/2, 1.5, 0],
        [-sqrt3, 0, 0],
        [-sqrt3/2, -1.5, 0],
        [sqrt3/2, -1.5, 0],
        [2.5*sqrt3, 0, 0],
        [-2.5*sqrt3, 0, 0],
        [sqrt3, 3, 0],
        [-sqrt3, -3, 0]
    ])
    
    init_outer = np.array([0, 0, 0])
    init_params = np.concatenate([init_inner.flatten(), init_outer])
    
    best_score = 0
    best_inner = init_inner
    best_outer = init_outer
    best_side = compute_outer_side(get_vertices_vectorized(init_inner), 0, 0, 0)
    
    # Check if initial is valid
    if check_overlaps(init_inner):
        best_score = 1.0 / best_side
    else:
        best_score = 0

    n_restarts = 100
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
            
        noise = np.random.normal(0, 0.5, size=36)
        params0 = init_params + noise
        
        try:
            res = minimize(objective, params0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            inner_res = res.x[:33].reshape(11, 3)
            Cx, Cy, alpha = res.x[33], res.x[34], res.x[35]
            
            if check_overlaps(inner_res):
                vertices = get_vertices_vectorized(inner_res)
                S = compute_outer_side(vertices, Cx, Cy, alpha)
                score = 1.0 / S
                if score > best_score:
                    best_score = score
                    best_inner = inner_res
                    best_outer = np.array([Cx, Cy, alpha])
                    best_side = S
                    
        except Exception:
            pass

    return best_inner, best_outer, best_side

hexagon_packing_11()
