import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Define the 11 hexagon centers in axial coordinates
    # This specific shape is a "radius 1" hexagon (7 hexes) plus 4 adjacent hexes
    # forming a compact cluster.
    coords_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    # Convert to Cartesian coordinates
    centers = []
    for q, r in coords_axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    
    # Outer hexagon normals (for rotation 0)
    angles = [np.deg2rad(a) for a in [30, 90, 150, 210, 270, 330]]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def objective(C):
        dx = centers - C
        projections = np.dot(dx, normals.T)
        abs_proj = np.abs(projections)
        apothem = np.sqrt(3) / 2.0
        vals = abs_proj + apothem
        return np.max(vals)
        
    # Initial guess: centroid
    C0 = np.mean(centers, axis=0)
    
    best_side = float('inf')
    best_C = C0
    
    # Try Nelder-Mead
    try:
        res = minimize(objective, C0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000})
        if res.success:
            best_C = res.x
            best_A = objective(best_C)
            best_side = 2.0 * best_A / np.sqrt(3)
    except Exception:
        pass
        
    # If optimization failed or is slow, use a simple grid search as backup
    if best_side == float('inf') or (time.time() - start_time > 10):
        # Grid search around centroid
        step = 0.1
        range_s = 2.0
        best_A = float('inf')
        for i in range(-20, 21):
            for j in range(-20, 21):
                C = C0 + np.array([i*step, j*step])
                A = objective(C)
                if A < best_A:
                    best_A = A
                    best_C = C
        best_side = 2.0 * best_A / np.sqrt(3)

    # Check time
    if time.time() - start_time > 140:
         # Return current best
         pass

    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    outer_center = (best_C[0], best_C[1])
    outer_side = best_side
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
