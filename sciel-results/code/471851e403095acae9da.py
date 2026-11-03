import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def compute_outer_side(inner_positions, outer_center, outer_angle):
    cx, cy = outer_center
    all_verts = []
    for (ix, iy, iang) in inner_positions:
        verts = hexagon_vertices(ix, iy, 1.0, iang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    angle_rad = np.radians(outer_angle)
    min_side = 0
    for vx, vy in all_verts:
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            val = (vx - cx) * np.cos(theta) + (vy - cy) * np.sin(theta)
            if val > min_side * np.sqrt(3) / 2:
                required_side = val / (np.sqrt(3) / 2)
                if required_side > min_side:
                    min_side = required_side
    return min_side

def objective(params):
    inner_positions = []
    for i in range(N):
        inner_positions.append((params[3*i], params[3*i+1], params[3*i+2]))
    outer_cx = params[3*N]
    outer_cy = params[3*N+1]
    outer_angle = params[3*N+2]
    return compute_outer_side(inner_positions, (outer_cx, outer_cy), outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def axial_to_pixel(q, r):
        x = np.sqrt(3) * (q + r/2)
        y = 3/2 * r
        return x, y
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (-1, 2), (0, 2)
    ]
    
    init_params = []
    for q, r in axial_coords:
        x, y = axial_to_pixel(q, r)
        init_params.extend([x, y, 0.0])
    init_params.extend([0.0, 0.0, 0.0])
    
    result = minimize(objective, init_params, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if time.time() > deadline:
        best_params = init_params
    else:
        best_params = result.x
    
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    outer_center = (best_params[3*N], best_params[3*N+1])
    outer_side = objective(best_params)
    outer_angle = best_params[3*N+2]
    
    return inner, outer_center, outer_side, outer_angle
