import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Directions to 6 vertices of a flat-top regular hexagon
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # 6x2
    
    def enclosing_hex_radius(points, c):
        diffs = points - c
        projs = diffs @ u.T
        return np.max(projs)
    
    def objective(c, points):
        return enclosing_hex_radius(points, c)
    
    # Hex lattice with spacing sqrt(3) for non-overlapping unit hexagons
    d = np.sqrt(3)
    v1 = np.array([d, 0.0])
    v2 = np.array([d / 2, d * np.sqrt(3) / 2])
    
    def hex_to_cart(q, r):
        return q * v1 + r * v2
    
    # Candidate arrangements of 11 hexagons in hex grid coordinates
    candidate_sets = [
        # Center + first ring (6) + 4 from second ring
        [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (2,0), (2,-1), (1,-2), (-1,-2)],
        [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (2,0), (2,-1), (2,-2), (1,-2)],
        [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (-2,0), (-2,1), (-1,2), (1,2)],
        # 3x3 block + 2
        [(q,r) for q in range(-1,2) for r in range(-1,2)] + [(2,0), (2,-1)],
        [(q,r) for q in range(-1,2) for r in range(-1,2)] + [(-2,0), (-2,1)],
        # 2x5 strip + 1
        [(q,r) for q in range(2) for r in range(-2,3)] + [(1,0)],
        [(q,r) for q in range(2) for r in range(-2,3)] + [(0,2)],
    ]
    
    best_side = float('inf')
    best_result = None
    
    for idx, hex_coords in enumerate(candidate_sets):
        if time.time() > deadline:
            break
        if len(hex_coords) != N:
            continue
        
        points = np.array([hex_to_cart(q, r) for q, r in hex_coords])
        c0 = np.mean(points, axis=0)
        
        try:
            result = minimize(objective, c0, args=(points,), method='Nelder-Mead',
                             options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
            R = result.fun
            if R < best_side:
                best_side = R
                best_result = (hex_coords, result.x, R)
        except Exception:
            continue
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    hex_coords, center, side = best_result
    inner = [[float(hex_to_cart(q, r)[0]), float(hex_to_cart(q, r)[1]), 0.0] for q, r in hex_coords]
    return inner, (float(center[0]), float(center[1])), float(side), 0.0
