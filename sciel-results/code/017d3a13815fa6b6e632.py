import numpy as np
from scipy.optimize import minimize
import time

N = 12
EPS = 1e-9

def hex_vertices(center, angle=0.0):
    """Returns vertices of a unit regular hexagon centered at 'center' with rotation 'angle'."""
    # Vertices of unit hexagon at (0,0), angle 0 (pointy top)
    # Angles: 30, 90, 150, 210, 270, 330 degrees?
    # Standard "pointy" hex: vertices at 30, 90, ...
    # Side length 1. Radius 1.
    angles = np.deg2rad(30 + 60 * np.arange(6)) + np.deg2rad(angle)
    return np.column_stack((center[0] + np.cos(angles), center[1] + np.sin(angles)))

def calc_L(centers, outer_angle_deg):
    """
    Calculate the minimal side length L of a regular hexagon (centered at 0,0, rotated by outer_angle_deg)
    that contains all vertices of the inner hexagons at 'centers' (aligned, angle 0).
    """
    # Normals for the outer hexagon
    # The boundary is defined by | n_k . p | <= L * sqrt(3) / 2
    # n_k are unit normals at 30 + outer_angle + 60*k
    outer_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = np.deg2rad(30) + outer_rad + k * np.deg2rad(60)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    max_proj = 0.0
    # Vertices of a unit hexagon at origin (aligned)
    base_angles = np.deg2rad(30 + 60 * np.arange(6))
    base_verts = np.column_stack((np.cos(base_angles), np.sin(base_angles)))
    
    for c in centers:
        verts = c + base_verts
        # Project all vertices onto all normals
        # verts shape (6,2), normals shape (6,2)
        # We need max | v . n | for all v in verts, n in normals
        # This is max over 36 values.
        # Can be done with matrix multiplication if we expand, but 12*6*6 is small.
        for v in verts:
            projs = np.abs(normals @ v)
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
    L = max_proj * 2.0 / np.sqrt(3)
    return L

def overlap_penalty(centers, min_dist=np.sqrt(3)):
    """
    Penalty for overlapping inner hexagons.
    Assumes aligned hexagons. Min distance between centers is sqrt(3).
    """
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return penalty

def objective(params):
    # params: [x1, y1, ..., x12, y12, angle]
    centers = params[:2*N].reshape(N, 2)
    angle = params[2*N]
    
    L = calc_L(centers, angle)
    pen = overlap_penalty(centers)
    
    return L + 1000.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess: 7-hex cluster + 5 extra
    # Lattice coords to Cartesian
    def lat_to_cart(i, j):
        x = np.sqrt(3) * i + np.sqrt(3)/2 * j
        y = 1.5 * j
        return [x, y]
        
    # 7 hexes
    coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # 5 extra
    coords += [(2,0), (1,1), (0,2), (-1,2), (-2,1)]
    
    centers_init = np.array([lat_to_cart(i,j) for i,j in coords])
    # Center the cluster
    centers_init -= np.mean(centers_init, axis=0)
    
    params_init = np.concatenate((centers_init.flatten(), [0.0]))
    
    best_score = np.inf
    best_params = params_init
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, params_init, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
        # Check time
        if time.time() > deadline:
            pass
        else:
            # Try a few random restarts if time permits
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                # Perturb initial guess
                noise = np.random.normal(0, 0.5, size=(N, 2))
                centers_pert = centers_init + noise
                centers_pert -= np.mean(centers_pert, axis=0)
                params_pert = np.concatenate((centers_pert.flatten(), [np.random.uniform(-30, 30)]))
                
                res2 = minimize(objective, params_pert, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x
                    
    except Exception as e:
        # Fallback to initial guess if optimization fails
        pass

    # Extract results
    centers = best_params[:2*N].reshape(N, 2)
    angle = best_params[2*N]
    
    # The outer hexagon is centered at (0,0) and rotated by 'angle'
    # The inner hexagons are at 'centers' with rotation 0
    # Return format:
    # inner: list of [x, y, angle_deg]
    # center: [cx, cy]
    # side: L
    # angle: outer_angle
    
    inner = [[c[0], c[1], 0.0] for c in centers]
    center = [0.0, 0.0]
    side = calc_L(centers, angle)
    # Note: The 'angle' in the return is the rotation of the OUTER hexagon.
    # In our objective, we optimized the outer hex angle relative to the fixed inner hexes.
    # So this is correct.
    
    return inner, center, side, angle

