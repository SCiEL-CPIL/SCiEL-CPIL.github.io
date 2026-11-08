import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Place 12 unit hexagons in a lattice configuration
    d = np.sqrt(3)  # center-to-center distance for touching unit hexagons
    
    positions = []
    # Row 0: 3 hexagons
    for j in range(3):
        positions.append(((j - 1) * d, 0.0))
    # Row 1: 4 hexagons
    for j in range(4):
        positions.append(((j - 1.5) * d, d * np.sqrt(3) / 2))
    # Row 2: 3 hexagons
    for j in range(3):
        positions.append(((j - 1) * d, d * np.sqrt(3)))
    # Row 3: 2 hexagons
    for j in range(2):
        positions.append(((j - 0.5) * d, d * np.sqrt(3) * 1.5))
    
    inner = [[p[0], p[1], 0.0] for p in positions]
    
    # Compute vertices of all inner hexagons
    all_vertices = []
    for x, y, angle_deg in inner:
        angle = np.radians(angle_deg)
        for k in range(6):
            v_angle = angle + k * np.pi / 3
            all_vertices.append((x + np.cos(v_angle), y + np.sin(v_angle)))
    
    all_vertices = np.array(all_vertices)
    
    def compute_s(c, theta):
        cx, cy = c
        normals = []
        for k in range(6):
            a = theta + k * np.pi / 3
            normals.append((np.cos(a), np.sin(a)))
        normals = np.array(normals)
        
        vc = all_vertices - np.array([cx, cy])
        projections = vc @ normals.T
        max_proj = np.max(projections, axis=0)
        s = np.max(max_proj) / (np.sqrt(3) / 2)
        return s
    
    def objective(params):
        cx, cy, theta = params
        return compute_s((cx, cy), theta)
    
    # Try multiple initial conditions
    best_s = float('inf')
    best_params = None
    
    for cx0 in [-1, 0, 1]:
        for cy0 in [-1, 0, 1]:
            for theta0 in [0, np.pi/6, np.pi/3]:
                x0 = [cx0, cy0, theta0]
                try:
                    result = minimize(objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                    s = result.fun
                    if s < best_s:
                        best_s = s
                        best_params = result.x
                except:
                    pass
    
    cx, cy, theta = best_params
    s = compute_s((cx, cy), theta)
    s = max(s, 1.0)
    theta_deg = np.degrees(theta)
    
    return inner, (cx, cy), s, theta_deg
