import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def hex_vertices(cx, cy, angle_deg):
    theta = np.radians(angle_deg)
    angles = theta + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def enclosing_hex_side(centers, angles_deg, outer_angle_deg):
    all_points = []
    for i, (cx, cy) in enumerate(centers):
        verts = hex_vertices(cx, cy, angles_deg[i])
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    theta = np.radians(outer_angle_deg)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    
    max_dist = 0
    for px, py in all_points:
        lx = px * cos_t - py * sin_t
        ly = px * sin_t + py * cos_t
        for k in range(6):
            angle_k = k * np.pi / 3
            nx, ny = np.cos(angle_k), np.sin(angle_k)
            dist = lx * nx + ly * ny
            if dist > max_dist:
                max_dist = dist
    
    return 2 * max_dist / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def axial_to_xy(q, r):
        x = 2.0 * q + r
        y = np.sqrt(3) * r
        return x, y
    
    hex_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2 and abs(q) <= 2 and abs(r) <= 2:
                x, y = axial_to_xy(q, r)
                hex_positions.append((x, y))
    
    hex_positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    initial_centers = hex_positions[:N]
    initial_angles = [0.0] * N
    outer_angle = 0.0
    
    centers_arr = np.array(initial_centers)
    current_side = enclosing_hex_side(initial_centers, initial_angles, outer_angle)
    
    best_centers = centers_arr.copy()
    best_angles = initial_angles.copy()
    best_outer_angle = outer_angle
    best_side = current_side
    
    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_ang = params[3*N]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1.99:
                    penalty += (1.99 - dist) ** 2 * 1000
        
        side = enclosing_hex_side(pos.tolist(), angles.tolist(), outer_ang)
        return side + penalty
    
    x0 = np.concatenate([centers_arr.flatten(), np.array(initial_angles), [outer_angle]])
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.fun < best_side:
        pos = result.x[:2*N].reshape(N, 2)
        angles = result.x[2*N:3*N].tolist()
        outer_ang = result.x[3*N]
        
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1.99:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best_centers = pos
            best_angles = angles
            best_outer_angle = outer_ang
            best_side = enclosing_hex_side(pos.tolist(), angles, outer_ang)
    
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            perturbed = best_centers + np.random.randn(N, 2) * 0.5
            perturbed_angles = [(a + np.random.randn() * 10) % 360 for a in best_angles]
            
            x0r = np.concatenate([perturbed.flatten(), np.array(perturbed_angles), [best_outer_angle + np.random.randn() * 10]])
            
            result_r = minimize(objective, x0r, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if result_r.fun < best_side:
                pos = result_r.x[:2*N].reshape(N, 2)
                angles = result_r.x[2*N:3*N].tolist()
                outer_ang = result_r.x[3*N]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1.99:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_centers = pos
                    best_angles = angles
                    best_outer_angle = outer_ang
                    best_side = enclosing_hex_side(pos.tolist(), angles, outer_ang)
    
    inner = [[float(best_centers[i,0]), float(best_centers[i,1]), float(best_angles[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), float(best_outer_angle)
# EVOLVE-BLOCK-END
