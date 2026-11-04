import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3)
    centers_init = np.array([
        [0, 0],
        [sqrt3, 0], [-sqrt3, 0],
        [0, sqrt3], [0, -sqrt3],
        [sqrt3/2, 3/2], [-sqrt3/2, 3/2],
        [sqrt3/2, -3/2], [-sqrt3/2, -3/2],
        [0, 2*sqrt3], [0, -2*sqrt3]
    ])
    
    x0 = centers_init.flatten()
    
    def get_verts(centers):
        angles = np.arange(6) * np.pi / 3
        verts_rel = np.column_stack([np.cos(angles), np.sin(angles)])
        verts = centers[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]
        return verts.reshape(-1, 2)
        
    def min_side(verts):
        # Grid search for optimal outer angle
        thetas = np.linspace(0, np.pi/3, 50, endpoint=False)
        min_r = np.inf
        for theta in thetas:
            angles = theta + np.arange(6) * np.pi / 3
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            proj = verts @ normals.T
            ranges = np.max(proj, axis=0) - np.min(proj, axis=0)
            r = np.max(ranges)
            if r < min_r:
                min_r = r
        return min_r / np.sqrt(3)
        
    def objective(x):
        c = x.reshape(N, 2)
        verts = get_verts(c)
        return min_side(verts)
        
    best_x = x0
    best_val = objective(x0)
    
    # Run Nelder-Mead
    while time.time() < deadline:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
            else:
                break
        except:
            break
            
    final_centers = best_x.reshape(N, 2)
    verts = get_verts(final_centers)
    
    # Final precise calculation of outer hexagon
    thetas = np.linspace(0, np.pi/3, 200, endpoint=False)
    best_theta = 0
    best_r = np.inf
    for theta in thetas:
        angles = theta + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        proj = verts @ normals.T
        ranges = np.max(proj, axis=0) - np.min(proj, axis=0)
        r = np.max(ranges)
        if r < best_r:
            best_r = r
            best_theta = theta
            
    optimal_side = best_r / np.sqrt(3)
    optimal_theta = best_theta
    
    angles = optimal_theta + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    proj = verts @ normals.T
    mids = (np.max(proj, axis=0) + np.min(proj, axis=0)) / 2
    center_outer = np.linalg.lstsq(normals, mids, rcond=None)[0]
    
    inner = [[c[0], c[1], 0.0] for c in final_centers]
    return inner, [center_outer[0], center_outer[1]], optimal_side, np.degrees(optimal_theta)

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will output the complete program.
# EVOLVE-BLOCK-END
