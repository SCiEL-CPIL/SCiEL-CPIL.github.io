import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    sqrt3 = np.sqrt(3)
    
    # Unit hexagon vertices (side 1, rotation 0, flat-top)
    hex_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def make_objective(cols, rows, shear_dir):
        """
        cols x rows grid, shear_dir: 'h' or 'v'
        """
        def objective(params):
            s, t, cx, cy, alpha = params
            
            # Lattice basis
            if shear_dir == 'h':
                a1 = np.array([sqrt3 * t, 0.0])
                a2_base = np.array([sqrt3/2 * t, 3/2 * t])
                a2 = a2_base + s * a1
            else:
                a1 = np.array([sqrt3 * t, 0.0])
                a2 = np.array([sqrt3/2 * t, 3/2 * t])
                # Vertical shear: modify a1
                a1 = a1 + s * a2
            
            centers = []
            for i in range(cols):
                for j in range(rows):
                    c = i * a1 + j * a2
                    centers.append(c)
            centers = np.array(centers)
            
            # Center at origin then shift
            com = np.mean(centers, axis=0)
            centers = centers - com + np.array([cx, cy])
            
            # All inner vertices
            all_verts = np.vstack([c + hex_verts for c in centers])
            
            # Outer hexagon edge normals
            alpha_rad = np.deg2rad(alpha)
            normal_angles = alpha_rad + np.deg2rad(30) + np.deg2rad(60) * np.arange(6)
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            oc = np.array([cx, cy])
            rel = all_verts - oc
            dots = rel @ normals.T
            max_dot = np.max(dots, axis=1)
            S_needed = np.max(max_dot) / (sqrt3 / 2.0)
            
            return max(S_needed, 0.0)
        
        return objective
    
    best_result = None
    best_params = None
    
    for cols, rows in [(3, 4), (4, 3)]:
        for shear_dir in ['h', 'v']:
            obj = make_objective(cols, rows, shear_dir)
            for x0 in [[0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.2, 1.0, 0.0, 0.0, 15.0],
                        [-0.2, 1.0, 0.0, 0.0, -15.0],
                        [0.1, 0.9, 0.0, 0.0, 30.0]]:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
                    best_params = (cols, rows, shear_dir, res.x)
    
    cols, rows, shear_dir, params = best_params
    s, t, cx, cy, alpha = params
    
    if shear_dir == 'h':
        a1 = np.array([sqrt3 * t, 0.0])
        a2_base = np.array([sqrt3/2 * t, 3/2 * t])
        a2 = a2_base + s * a1
    else:
        a1 = np.array([sqrt3 * t, 0.0])
        a2 = np.array([sqrt3/2 * t, 3/2 * t])
        a1 = a1 + s * a2
    
    centers = []
    for i in range(cols):
        for j in range(rows):
            c = i * a1 + j * a2
            centers.append(c)
    centers = np.array(centers)
    com = np.mean(centers, axis=0)
    centers = centers - com + np.array([cx, cy])
    
    inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
    outer_center = (float(cx), float(cy))
    outer_side = float(best_result.fun)
    outer_angle = float(alpha)
    
    return inner, outer_center, outer_side, outer_angle
