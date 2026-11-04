import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate 12 inner hexagon centers
    # Flat-top hexagons tiling
    # v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # We use a 3x4 grid
    v1 = np.array([1.5, np.sqrt(3)/2])
    v2 = np.array([0.0, np.sqrt(3)])
    
    centers = []
    for i in range(4):
        for j in range(3):
            c = i * v1 + j * v2
            centers.append(c)
            
    centers = np.array(centers)
    
    # 2. Generate vertices of all inner hexagons
    # Unit hexagon vertices (flat top)
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    verts_unit = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_verts = []
    for c in centers:
        all_verts.append(c + verts_unit)
        
    all_verts = np.vstack(all_verts)
    
    # Center the point cloud
    centroid = np.mean(all_verts, axis=0)
    all_verts_centered = all_verts - centroid
    
    # 3. Define objective function
    def objective(alpha):
        # alpha is in radians
        # Normals of the outer hexagon
        u1 = np.array([np.cos(alpha), np.sin(alpha)])
        u2 = np.array([np.cos(alpha + 2*np.pi/3), np.sin(alpha + 2*np.pi/3)])
        u3 = np.array([np.cos(alpha + 4*np.pi/3), np.sin(alpha + 4*np.pi/3)])
        
        # Project points
        p1 = all_verts_centered @ u1
        p2 = all_verts_centered @ u2
        p3 = all_verts_centered @ u3
        
        w1 = np.max(p1) - np.min(p1)
        w2 = np.max(p2) - np.min(p2)
        w3 = np.max(p3) - np.min(p3)
        
        return max(w1, w2, w3)
        
    # 4. Optimize alpha
    # Coarse grid search
    best_alpha = 0.0
    best_val = np.inf
    
    # Search in [0, 60) degrees due to symmetry
    alphas = np.linspace(0, np.pi/3, 1000)
    for a in alphas:
        val = objective(a)
        if val < best_val:
            best_val = val
            best_alpha = a
            
    # Fine polish
    bounds = (best_alpha - 0.01, best_alpha + 0.01)
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-8})
    
    if res.success:
        best_alpha = res.x
        best_val = res.fun
        
    # 5. Calculate side length and center
    S = best_val / np.sqrt(3)
    
    u1 = np.array([np.cos(best_alpha), np.sin(best_alpha)])
    u2 = np.array([np.cos(best_alpha + 2*np.pi/3), np.sin(best_alpha + 2*np.pi/3)])
    
    p1 = all_verts_centered @ u1
    p2 = all_verts_centered @ u2
    
    m1 = (np.max(p1) + np.min(p1)) / 2.0
    m2 = (np.max(p2) + np.min(p2)) / 2.0
    
    A = np.array([
        [u1[0], u1[1]],
        [u2[0], u2[1]]
    ])
    b = np.array([m1, m2])
    
    try:
        C_rel = np.linalg.solve(A, b)
    except:
        C_rel = np.array([0.0, 0.0])
        
    C_actual = centroid + C_rel
    
    # 6. Construct return values
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    center = [C_actual[0], C_actual[1]]
    side = S
    angle_deg = np.rad2deg(best_alpha)
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END
