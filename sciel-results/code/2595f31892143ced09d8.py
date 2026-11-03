import math
import time
import numpy as np
from scipy.optimize import linprog

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3)
    
    # Place 12 unit hexagons in a compact hexagonal lattice (3 cols x 4 rows)
    positions = []
    for col in range(3):
        for row in range(4):
            x = col * 1.5
            y = row * sqrt3 + (col % 2) * (sqrt3 / 2)
            positions.append((x, y, 0.0))
    
    inner = [[x, y, a] for x, y, a in positions]
    
    # Get all vertices
    all_verts = []
    for x, y, a in inner:
        verts = hex_vertices(x, y, a)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    def solve_for_theta(theta_deg):
        if time.time() > deadline:
            return np.inf, (0, 0)
        theta = math.radians(theta_deg)
        normals = [(math.cos(theta + k * math.pi / 3), math.sin(theta + k * math.pi / 3)) for k in range(6)]
        
        A_ub = []
        b_ub = []
        for px, py in all_verts:
            for cos_phi, sin_phi in normals:
                A_ub.append([-cos_phi, -sin_phi, -sqrt3 / 2])
                b_ub.append(-px * cos_phi - py * sin_phi)
        
        c = [0, 0, 1]
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        
        try:
            res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
            if res.success:
                return res.fun, (res.x[0], res.x[1])
        except:
            pass
        return np.inf, (0, 0)
    
    best_s = np.inf
    best_theta = 0.0
    best_center = (0, 0)
    
    # Coarse search
    for theta_deg in range(0, 180, 3):
        s, center = solve_for_theta(theta_deg)
        if s < best_s:
            best_s = s
            best_theta = theta_deg
            best_center = center
    
    # Fine search around best
    for theta_deg in np.arange(best_theta - 3, best_theta + 3.1, 0.5):
        s, center = solve_for_theta(theta_deg)
        if s < best_s:
            best_s = s
            best_theta = theta_deg
            best_center = center
    
    return inner, best_center, best_s, best_theta
