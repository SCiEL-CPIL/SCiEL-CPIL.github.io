import numpy as np
import scipy.optimize as opt
import time

N = 11

def construct():
    # Unit hexagon vertices
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    base_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Initial configuration: 3-5-3 lattice
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    row2 = [i * v1 for i in range(-2, 3)]
    row1 = [(i * v1 + v2) for i in range(-1, 2)]
    row3 = [(i * v1 - v2) for i in range(-1, 2)]
    
    points = row1 + row2 + row3
    centroid = np.mean(points, axis=0)
    points = [p - centroid for p in points]
    
    x0 = np.array([p[0] for p in points])
    y0 = np.array([p[1] for p in points])
    theta0 = np.zeros(N)
    p0 = np.concatenate([x0, y0, theta0, [0.0]])
    
    def get_S_and_penalty(p):
        x = p[0:N]
        y = p[N:2*N]
        theta = p[2*N:3*N]
        phi = p[3*N]
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < np.sqrt(3):
                    penalty += (np.sqrt(3) - dist)**2
        
        # Calculate S
        max_proj = 0.0
        for k in range(6):
            angle_n = phi + np.pi/6 + k * np.pi/3
            nx = np.cos(angle_n)
            ny = np.sin(angle_n)
            for i in range(N):
                cos_t = np.cos(theta[i])
                sin_t = np.sin(theta[i])
                # Rotation
                vx = base_vertices[:, 0] * cos_t - base_vertices[:, 1] * sin_t
                vy = base_vertices[:, 0] * sin_t + base_vertices[:, 1] * cos_t
                vx += x[i]
                vy += y[i]
                projs = vx * nx + vy * ny
                m = np.max(projs)
                if m > max_proj:
                    max_proj = m
        
        S = (2.0 / np.sqrt(3)) * max_proj
        return S, penalty

    def cost_func(p):
        S, pen = get_S_and_penalty(p)
        return S + 100.0 * pen

    best_S = float('inf')
    best_p = p0
    best_valid_S = float('inf')
    best_valid_p = p0
    
    # Check initial
    S_init, pen_init = get_S_and_penalty(p0)
    if pen_init < 1e-6:
        best_valid_S = S_init
        best_valid_p = p0
        best_S = S_init
    
    # Optimize
    try:
        res = opt.minimize(cost_func, p0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        S_res, pen_res = get_S_and_penalty(res.x)
        if pen_res < 1e-6 and S_res < best_valid_S:
            best_valid_S = S_res
            best_valid_p = res.x
    except Exception:
        pass
    
    # Use the best valid result
    x_best = best_valid_p[0:N]
    y_best = best_valid_p[N:2*N]
    theta_best = best_valid_p[2*N:3*N]
    phi_best = best_valid_p[3*N]
    
    S_final, _ = get_S_and_penalty(best_valid_p)
    
    inner = []
    for i in range(N):
        inner.append([x_best[i], y_best[i], np.degrees(theta_best[i])])
    
    return inner, (0.0, 0.0), S_final, np.degrees(phi_best)
