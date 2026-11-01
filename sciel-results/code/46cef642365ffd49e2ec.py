import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    # Minimum center-to-center distance for two non-overlapping unit hexagons (same orientation)
    # = inradius of Minkowski sum = 2 * (sqrt(3)/2) = sqrt(3)
    d_min = np.sqrt(3)
    
    # Hexagonal lattice: axial coordinates to cartesian
    # For flat-topped hex grid with spacing d_min:
    # x = d_min * (q + r/2), y = d_min * (sqrt(3)/2) * r
    def axial_to_cart(q, r, d):
        x = d * (q + r / 2.0)
        y = d * (np.sqrt(3) / 2.0) * r
        return x, y
    
    # Try several cluster configurations for 11 hexagons
    configurations = [
        # Config 1: center + ring6 + 4 from ring2
        [(0,0), (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1), (2,0),(-2,0),(0,2),(0,-2)],
        # Config 2: 3-4-3-1 rows
        [(0,0),(1,0),(2,0), (-1,1),(0,1),(1,1),(2,1), (-1,2),(0,2),(1,2), (0,3)],
        # Config 3: 4-3-4 
        [(0,0),(1,0),(2,0),(3,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2),(3,2)],
        # Config 4: 2-3-3-2-1
        [(0,0),(1,0), (0,1),(1,1),(2,1), (0,2),(1,2),(2,2), (0,3),(1,3), (0,4)],
        # Config 5: 1-2-3-2-1-2 (more spread)
        [(0,0), (0,1),(1,1), (0,2),(1,2),(2,2), (0,3),(1,3), (0,4),(1,4), (0,5)],
    ]
    
    def max_projection(points, center, alpha_deg):
        alpha = np.radians(alpha_deg)
        normals = np.array([[np.cos(alpha + np.radians(60 * i)), 
                            np.sin(alpha + np.radians(60 * i))] for i in range(6)])
        rel = points - center
        projs = normals @ rel.T
        return np.max(projs)
    
    def required_side(positions, center, alpha_deg):
        mp = max_projection(positions, center, alpha_deg)
        return mp * 2.0 / np.sqrt(3)
    
    def objective(params, positions):
        cx, cy, alpha = params
        return required_side(positions, np.array([cx, cy]), alpha)
    
    best_side = float('inf')
    best_config = None
    best_params = None
    
    for config in configurations:
        if time.time() > deadline:
            break
        positions = np.array([axial_to_cart(q, r, d_min) for q, r in config])
        com = np.mean(positions, axis=0)
        
        for start_alpha in [0, 15, 30, 45, 60, 90]:
            if time.time() > deadline:
                break
            x0 = [com[0], com[1], float(start_alpha)]
            try:
                res = minimize(lambda p: objective(p, positions), x0, 
                             method='Nelder-Mead', 
                             options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < best_side:
                    best_side = res.fun
                    best_config = config
                    best_params = res.x
            except:
                pass
    
    if best_config is None:
        # Fallback
        config = configurations[0]
        positions = np.array([axial_to_cart(q, r, d_min) for q, r in config])
        com = np.mean(positions, axis=0)
        best_params = [com[0], com[1], 0.0]
        best_side = required_side(positions, com, 0.0)
    
    cx, cy, alpha = best_params
    positions = np.array([axial_to_cart(q, r, d_min) for q, r in best_config])
    final_side = required_side(positions, np.array([cx, cy]), alpha)
    
    inner = [[float(positions[i][0]), float(positions[i][1]), 0.0] for i in range(N)]
    
    return inner, (float(cx), float(cy)), float(final_side), float(alpha)
# EVOLVE-BLOCK-END
