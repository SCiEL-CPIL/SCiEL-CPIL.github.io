import numpy as np
import time
from scipy.optimize import minimize

N = 12

def get_hexagon_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def polygons_overlap(poly1, poly2):
    n1 = len(poly1)
    n2 = len(poly2)
    
    def project(points, axis):
        return np.dot(points, axis)
    
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
        return axes
    
    axes = get_axes(poly1) + get_axes(poly2)
    
    for axis in axes:
        proj1 = project(poly1, axis)
        proj2 = project(poly2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def calculate_min_L(positions, rotations, theta_container):
    max_proj = 0.0
    for i in range(N):
        cx, cy = positions[i]
        ang = rotations[i]
        verts = get_hexagon_vertices(cx, cy, ang)
        
        for v in verts:
            for k in range(6):
                theta_n = np.deg2rad(theta_container) + k * np.pi / 3.0
                nx = np.cos(theta_n)
                ny = np.sin(theta_n)
                proj = v[0] * nx + v[1] * ny
                if proj > max_proj:
                    max_proj = proj
                    
    L = max_proj / (np.sqrt(3) / 2.0)
    return L

best_valid_params = None
best_valid_L = float('inf')
start_time = 0.0

def objective(params):
    global best_valid_params, best_valid_L, start_time
    
    if time.time() - start_time > 140:
        return 1e6
        
    positions = np.zeros((N, 2))
    rotations = np.zeros(N)
    
    for i in range(N):
        positions[i, 0] = params[i*3]
        positions[i, 1] = params[i*3 + 1]
        rotations[i] = params[i*3 + 2]
        
    theta_container = params[N*3]
    
    L = calculate_min_L(positions, rotations, theta_container)
    
    overlap_count = 0
    for i in range(N):
        poly_i = get_hexagon_vertices(positions[i, 0], positions[i, 1], rotations[i])
        for j in range(i+1, N):
            poly_j = get_hexagon_vertices(positions[j, 0], positions[j, 1], rotations[j])
            if polygons_overlap(poly_i, poly_j):
                overlap_count += 1
                
    if overlap_count == 0:
        if L < best_valid_L:
            best_valid_L = L
            best_valid_params = params.copy()
            
    return L + 100.0 * overlap_count

def construct():
    global start_time, best_valid_params, best_valid_L
    
    start_time = time.time()
    best_valid_params = None
    best_valid_L = float('inf')
    
    # Initial guess
    params = np.zeros(37)
    params[0] = 0.0
    params[1] = 0.0
    params[2] = 0.0
    
    for k in range(6):
        idx = (k+1)*3
        angle = k * 60.0
        dist = np.sqrt(3.0)
        params[idx] = dist * np.cos(np.deg2rad(angle))
        params[idx+1] = dist * np.sin(np.deg2rad(angle))
        params[idx+2] = 0.0
        
    for k in range(5):
        idx = (k+7)*3
        angle = 30.0 + k * 60.0
        dist = 2.0 * np.sqrt(3.0)
        params[idx] = dist * np.cos(np.deg2rad(angle))
        params[idx+1] = dist * np.sin(np.deg2rad(angle))
        params[idx+2] = 0.0
        
    params[36] = 0.0
    
    # Evaluate initial guess
    init_score = objective(params)
    
    try:
        result = minimize(objective, params, method='Powell', options={'maxiter': 2000, 'disp': False})
    except Exception:
        pass
        
    if best_valid_params is None:
        # Fallback to initial if no valid solution found (should not happen)
        best_valid_params = params
        best_valid_L = calculate_min_L(
            np.array([[params[i*3], params[i*3+1]] for i in range(N)]),
            np.array([params[i*3+2] for i in range(N)]),
            params[36]
        )
        
    positions = np.zeros((N, 2))
    rotations = np.zeros(N)
    for i in range(N):
        positions[i, 0] = best_valid_params[i*3]
        positions[i, 1] = best_valid_params[i*3 + 1]
        rotations[i] = best_valid_params[i*3 + 2]
    theta_container = best_valid_params[N*3]
    
    L = best_valid_L
    
    inner = []
    for i in range(N):
        inner.append([positions[i, 0], positions[i, 1], rotations[i]])
        
    return inner, (0.0, 0.0), L, theta_container
