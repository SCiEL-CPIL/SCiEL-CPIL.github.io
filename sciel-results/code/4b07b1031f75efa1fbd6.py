import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0 # 5s margin
    
    sqrt3 = np.sqrt(3)
    u = np.array([sqrt3, 0])
    v = np.array([sqrt3/2, 1.5])
    
    def lattice_to_cart(i, j):
        return i * u + j * v

    core = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Generate candidates
    candidates = []
    candidates.append(core + [(2,0), (-2,0), (0,2), (0,-2)])
    candidates.append(core + [(1,1), (-1,-1), (2,0), (-2,0)])
    candidates.append(core + [(2,0), (2,-1), (-2,1), (-2,0)])
    candidates.append(core + [(0,2), (0,-2), (1,1), (-1,-1)])
    candidates.append(core + [(2,0), (-2,0), (1,1), (-1,-1)])
    candidates.append(core + [(2,0), (2,-1), (1,1), (-1,-1)])
    
    def get_hex_vertices(theta):
        angles = np.deg2rad(np.arange(0, 360, 60) + theta)
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def calc_min_side(points_cart, theta):
        verts = get_hex_vertices(theta)
        
        c_rot = np.column_stack((
            points_cart[:, 0] * np.cos(theta) - points_cart[:, 1] * np.sin(theta),
            points_cart[:, 0] * np.sin(theta) + points_cart[:, 1] * np.cos(theta)
        ))
        
        v_rot = np.column_stack((
            verts[:, 0] * np.cos(theta) - verts[:, 1] * np.sin(theta),
            verts[:, 0] * np.sin(theta) + verts[:, 1] * np.cos(theta)
        ))
        
        all_verts = c_rot[:, np.newaxis, :] + v_rot[np.newaxis, :, :]
        all_verts = all_verts.reshape(-1, 2)
        
        phi = np.deg2rad(np.arange(30, 360, 60))
        normals = np.column_stack((np.cos(phi), np.sin(phi)))
        
        projections = all_verts @ normals.T
        max_proj = np.max(projections)
        
        return 2 * max_proj / sqrt3

    def optimize_cluster(points_lattice):
        points_cart = np.array([lattice_to_cart(i, j) for i, j in points_lattice])
        
        # Grid search
        thetas = np.linspace(0, np.pi/3, 2000)
        best_s = np.inf
        best_theta = 0
        
        for th in thetas:
            s = calc_min_side(points_cart, th)
            if s < best_s:
                best_s = s
                best_theta = th
                
        # Refine
        try:
            res = minimize_scalar(lambda th: calc_min_side(points_cart, th), 
                                  bounds=(max(0, best_theta - 0.005), min(np.pi/3, best_theta + 0.005)), 
                                  method='bounded', options={'xatol': 1e-7})
            if res.success and res.fun < best_s:
                best_s = res.fun
                best_theta = res.x
        except:
            pass
            
        return best_s, best_theta, points_cart

    best_overall_s = np.inf
    best_result = None
    
    for cand in candidates:
        if time.time() > deadline:
            break
            
        s, theta, points_cart = optimize_cluster(cand)
        
        if s < best_overall_s:
            best_overall_s = s
            angle_deg = np.rad2deg(theta)
            inner = []
            for p in points_cart:
                inner.append([p[0], p[1], angle_deg])
            
            best_result = (inner, (0.0, 0.0), best_overall_s, 0.0)
            
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    return best_result
# EVOLVE-BLOCK-END
