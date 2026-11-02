"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    sqrt3 = np.sqrt(3.0)
    
    # 11 hexagons in compact hexagonal cluster (axial coords, pointy-top)
    axial_positions = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-2, 1), (-1, -1)
    ]
    
    inner_pts = np.array([[sqrt3 * (q + r/2.0), 1.5 * r] for q, r in axial_positions])
    
    def min_hex_side(pts, center, theta):
        """Min side length of regular hexagon (center, theta) containing all pts."""
        cx, cy = center
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        
        max_proj = 0.0
        for i in range(6):
            angle = i * np.pi / 3.0
            nx = np.cos(angle)
            ny = np.sin(angle)
            projs = rx * nx + ry * ny
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
        
        return 2.0 * max_proj / sqrt3
    
    # Coarse search
    xs, ys = inner_pts[:, 0], inner_pts[:, 1]
    cx_vals = np.linspace(xs.min() - 0.5, xs.max() + 0.5, 40)
    cy_vals = np.linspace(ys.min() - 0.5, ys.max() + 0.5, 40)
    th_vals = np.linspace(0, np.pi/6, 15)
    
    best_s = np.inf
    best_c = (0.0, 0.0)
    best_th = 0.0
    
    for cx in cx_vals:
        for cy in cy_vals:
            for th in th_vals:
                if time.time() > deadline - 10:
                    break
                s = min_hex_side(inner_pts, (cx, cy), th)
                if s < best_s:
                    best_s = s
                    best_c = (cx, cy)
                    best_th = th
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Fine search
    if time.time() < deadline - 10:
        cx_f = np.linspace(best_c[0] - 0.3, best_c[0] + 0.3, 25)
        cy_f = np.linspace(best_c[1] - 0.3, best_c[1] + 0.3, 25)
        th_f = np.linspace(best_th - 0.2, best_th + 0.2, 25)
        
        for cx in cx_f:
            for cy in cy_f:
                for th in th_f:
                    if time.time() > deadline - 5:
                        break
                    s = min_hex_side(inner_pts, (cx, cy), th)
                    if s < best_s:
                        best_s = s
                        best_c = (cx, cy)
                        best_th = th
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
    
    inner_list = [[float(p[0]), float(p[1]), 0.0] for p in inner_pts]
    return inner_list, (float(best_c[0]), float(best_c[1])), float(best_s), float(best_th)
# EVOLVE-BLOCK-END
