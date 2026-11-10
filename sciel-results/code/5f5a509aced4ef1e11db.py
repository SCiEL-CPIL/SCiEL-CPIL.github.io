import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0
    
    def get_hex_vertices(cx, cy, s, angle_deg):
        a = np.radians(angle_deg)
        angles = a + np.radians(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def get_all_inner_vertices(positions):
        verts = []
        for i in range(len(positions)):
            cx, cy = positions[i]
            a = np.radians(0)  # angle = 0
            v_angles = a + np.radians(np.arange(6) * 60)
            v = np.column_stack([cx + np.cos(v_angles), cy + np.sin(v_angles)])
            verts.append(v)
        return np.vstack(verts)
    
    def min_containing_hex(vertices, angle_deg=0):
        """
        Returns (s, center) for minimal regular hexagon with given angle containing all vertices.
        """
        a = np.radians(angle_deg)
        normal_angles = a + np.radians(30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        proj = vertices @ normals.T  # (N, 6)
        proj_max = proj.max(axis=0)
        proj_min = proj.min(axis=0)
        widths = proj_max - proj_min
        
        s = np.max(widths) / np.sqrt(3)
        
        # Find center: c·n_k = (proj_max[k] + proj_min[k])/2 for all k
        # This is overdetermined; use least squares
        midpoints = (proj_max + proj_min) / 2
        c, _, _, _ = np.linalg.lstsq(normals.T, midpoints, rcond=None)
        
        return s, c
    
    def check_separation(positions):
        """
        Check if all pairs of unit hexagons are separated.
        Returns total violation (0 if all separated).
        """
        total_violation = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                vi = get_hex_vertices(positions[i][0], positions[i][1], 1.0, 0.0)
                vj = get_hex_vertices(positions[j][0], positions[j][1], 1.0, 0.0)
                
                # Get edge normals
                def get_edge_normals(verts):
                    normals = []
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k + 1) % 6]
                        edge = p2 - p1
                        n = np.array([-edge[1], edge[0]])
                        n = n / np.linalg.norm(n)
                        normals.append(n)
                    return np.array(normals)
                
                all_normals = np.vstack([get_edge_normals(vi), get_edge_normals(vj)])
                
                # SAT: find if any normal separates them
                min_overlap = 0.0  # most negative gap (largest overlap)
                for n in all_normals:
                    pi = vi @ n
                    pj = vj @ n
                    gap = min(pj.min() - pi.max(), pi.min() - pj.max())
                    # gap > 0 means separated, gap < 0 means overlapping
                    if gap < min_overlap:
                        min_overlap = gap
                
                if min_overlap < 0:
                    total_violation += -min_overlap
        
        return total_violation
    
    def objective(x):
        positions = x.reshape(12, 2)
        all_verts = get_all_inner_vertices(positions)
        s, _ = min_containing_hex(all_verts, angle_deg=0)
        
        sep_viol = check_separation(positions)
        
        if sep_viol > 1e-8:
            return s + 1000.0 * sep_viol
        return s
    
    # Initial arrangement: hexagonal cluster of 12
    sqrt3 = np.sqrt(3)
    # 1 center + 6 ring + 5 outer
    initial = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, sqrt3],
        [-1.0, sqrt3],
        [-2.0, 0.0],
        [-1.0, -sqrt3],
        [1.0, -sqrt3],
        [3.0, sqrt3],
        [0.0, 2.0 * sqrt3],
        [-3.0, sqrt3],
        [-3.0, -sqrt3],
        [0.0, -2.0 * sqrt3],
    ])
    
    x0 = initial.flatten()
    
    best_s = objective(x0)
    best_x = x0.copy()
    
    # Multiple Nelder-Mead restarts
    np.random.seed(123)
    
    for restart in range(8):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(24) * 0.3
        
        try:
            res = minimize(objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_s:
                best_s = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Final polish
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_s:
                best_s = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    positions = best_x.reshape(12, 2)
    all_verts = get_all_inner_vertices(positions)
    s, center = min_containing_hex(all_verts, angle_deg=0)
    
    # Small margin for numerical safety
    s *= 1.00005
    
    inner_hex_data = np.column_stack([positions[:, 0], positions[:, 1], np.zeros(12)])
    outer_hex_data = np.array([center[0], center[1], 0.0])
    
    return inner_hex_data, outer_hex_data, s

