import numpy as np
from scipy.optimize import minimize
import math

def hexagon_packing_12():
    n = 12
    sqrt3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, d, angle_deg):
        angle_rad = math.radians(angle_deg)
        return np.array([[cx + d * math.cos(angle_rad + math.radians(60*i)),
                          cy + d * math.sin(angle_rad + math.radians(60*i))]
                         for i in range(6)])
    
    def sat_overlap(verts1, verts2):
        normals = []
        for verts in [verts1, verts2]:
            for i in range(6):
                p1, p2 = verts[i], verts[(i+1)%6]
                edge = p2 - p1
                normal = np.array([edge[1], -edge[0]])
                nl = np.linalg.norm(normal)
                if nl > 1e-12:
                    normals.append(normal / nl)
        
        min_overlap = float('inf')
        for normal in normals:
            proj1 = verts1 @ normal
            proj2 = verts2 @ normal
            overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
            min_overlap = min(min_overlap, overlap)
        
        return min_overlap
    
    def max_vertex_projection(centers, outer_center):
        shifted = centers - outer_center
        max_proj = 0.0
        for c in shifted:
            for i in range(6):
                vx = c[0] + math.cos(math.radians(60*i))
                vy = c[1] + math.sin(math.radians(60*i))
                for j in range(6):
                    nx = math.cos(math.radians(30+60*j))
                    ny = math.sin(math.radians(30+60*j))
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        return max_proj
    
    def objective(params):
        centers = params.reshape(12, 2)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                v1 = get_hex_vertices(centers[i,0], centers[i,1], 1, 0)
                v2 = get_hex_vertices(centers[j,0], centers[j,1], 1, 0)
                overlap = sat_overlap(v1, v2)
                if overlap > 0:
                    penalty += overlap * overlap
        
        if penalty > 1e-8:
            return 1e6 + penalty
        
        centroid = np.mean(centers, axis=0)
        apothem = max_vertex_projection(centers, centroid)
        s = 2.0 * apothem / sqrt3
        if s < math.sqrt(12):
            s = math.sqrt(12)
        
        return s
    
    # Initial hex grid
    positions = []
    for r in [-2, -1, 0, 1]:
        for q in range(-3, 4):
            if len(positions) < 12:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                positions.append([x, y])
    
    positions = np.array(positions[:12], dtype=float)
    centroid = np.mean(positions, axis=0)
    positions = positions - centroid
    
    result = minimize(objective, positions.flatten(), method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    centers = result.x.reshape(12, 2)
    centroid = np.mean(centers, axis=0)
    apothem = max_vertex_projection(centers, centroid)
    s = max(2.0 * apothem / sqrt3, math.sqrt(12))
    
    inner = np.zeros((12, 3))
    inner[:, 0] = centers[:, 0] + centroid[0]
    inner[:, 1] = centers[:, 1] + centroid[1]
    inner[:, 2] = 0.0
    
    outer = np.array([centroid[0], centroid[1], 0.0])
    
    return inner, outer, s
