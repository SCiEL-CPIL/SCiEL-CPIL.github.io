import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    a = np.sqrt(3)  # center-to-center distance for side-to-side touching
    
    def hex_to_xy(q, r):
        x = a * (q + r / 2.0)
        y = a * (3.0 / 2.0) * r
        return x, y
    
    # 11 hexagons in compact arrangement
    positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-2, 0),
    ]
    
    centers = []
    for q, r in positions_axial:
        x, y = hex_to_xy(q, r)
        centers.append([x, y, 0.0])
    
    inner = centers
    centers_xy = np.array([[c[0], c[1]] for c in inner])
    
    def enclosing_hexagon_size(centers_xy, cx, cy, theta):
        max_proj = 0.0
        for i in range(len(centers_xy)):
            dx = centers_xy[i, 0] - cx
            dy = centers_xy[i, 1] - cy
            for k in range(6):
                normal_angle = theta + np.pi/6.0 + k * np.pi / 3.0
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                proj = dx * nx + dy * ny + 1.0
                if proj > max_proj:
                    max_proj = proj
        return max_proj * 2.0 / np.sqrt(3.0)
    
    def objective(params):
        cx, cy, theta = params
        return enclosing_hexagon_size(centers_xy, cx, cy, theta)
    
    cx0 = np.mean(centers_xy[:, 0])
    cy0 = np.mean(centers_xy[:, 1])
    
    best_size = float('inf')
    best_params = None
    
    initial_thetas = [0.0, np.pi/6, np.pi/3, np.pi/12, np.pi/4, np.pi/2, np.pi/5, 2*np.pi/3]
    
    for theta0 in initial_thetas:
        if time.time() > deadline:
            break
        x0 = [cx0, cy0, theta0]
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_size:
                best_size = res.fun
                best_params = res.x
        except:
            pass
    
    if best_params is not None:
        cx, cy, theta = best_params
        side = best_size
    else:
        cx, cy, theta = cx0, cy0, 0.0
        side = enclosing_hexagon_size(centers_xy, cx, cy, theta)
    
    angle_degrees = np.degrees(theta)
    
    return inner, (float(cx), float(cy)), float(side), float(angle_degrees)
# EVOLVE-BLOCK-END
