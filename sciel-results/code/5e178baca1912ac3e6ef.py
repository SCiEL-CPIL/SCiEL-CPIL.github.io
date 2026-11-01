import numpy as np
from scipy.optimize import minimize_scalar

def hexagon_vertices(center, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = center[0] + side * np.cos(theta)
        y = center[1] + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def min_enclosing_hexagon_side(vertices, center=(0,0)):
    def side_length(theta_deg):
        max_proj = 0
        for p in vertices:
            p_rel = p - np.array(center)
            for k in range(6):
                theta = np.deg2rad(theta_deg) + np.pi/6 + k * np.pi/3
                n = np.array([np.cos(theta), np.sin(theta)])
                proj = np.dot(p_rel, n)
                if proj > max_proj:
                    max_proj = proj
        return max_proj * 2 / np.sqrt(3)
    
    res = minimize_scalar(side_length, bounds=(0, 180), method='bounded', options={'xatol': 1e-4})
    return res.fun

def construct():
    N = 11
    v1 = np.array([np.sqrt(3), 0])
    v2 = np.array([np.sqrt(3)/2, 1.5])
    
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.linalg.norm(p) < 5:
                lattice_points.append(p)
    
    lattice_points.sort(key=lambda p: np.linalg.norm(p))
    selected = lattice_points[:11]
    
    all_vertices = []
    for c in selected:
        verts = hexagon_vertices(c, 1.0, 0.0)
        all_vertices.extend(verts.tolist())
    all_vertices = np.array(all_vertices)
    
    side = min_enclosing_hexagon_side(all_vertices)
    
    inner = [[c[0], c[1], 0.0] for c in selected]
    return inner, (0.0, 0.0), side, 0.0
