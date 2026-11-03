import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return np.array(verts)

    def compute_min_side(points, angle_deg):
        a = math.radians(angle_deg)
        normals = []
        for k in range(3):
            theta = a + math.pi/6 + k * math.pi/3
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        max_r = 0
        for n in normals:
            proj = points @ n
            min_p = np.min(proj)
            max_p = np.max(proj)
            r = (max_p - min_p) / 2.0
            if r > max_r:
                max_r = r
                
        side = 2.0 * max_r / math.sqrt(3)
        return side

    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    indices = [
        (0,0),
        (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1),
        (2,0), (-2,0), (0,2), (0,-2), (1,1)
    ]
    
    scale = 1.000001
    centers = []
    for i, j in indices:
        c = (i * u + j * v) * scale
        centers.append(c)
        
    inner_angle = 90.0
    
    all_verts = []
    for c in centers:
        verts = hex_vertices(c[0], c[1], inner_angle, 1.0)
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_angle = 0.0
    
    for a in np.arange(0, 60, 0.5):
        side = compute_min_side(all_verts, a)
        if side < best_side:
            best_side = side
            best_angle = a
            
    for a in np.arange(best_angle - 0.5, best_angle + 0.5, 0.01):
        side = compute_min_side(all_verts, a)
        if side < best_side:
            best_side = side
            best_angle = a
            
    a = math.radians(best_angle)
    normals = []
    for k in range(3):
        theta = a + math.pi/6 + k * math.pi/3
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
    mids = []
    for n in normals:
        proj = all_verts @ n
        mids.append((np.min(proj) + np.max(proj)) / 2.0)
        
    A = np.array(normals)
    b = np.array(mids)
    
    C, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    best_center = (C[0], C[1])
    
    inner = []
    for c in centers:
        inner.append([c[0], c[1], inner_angle])
        
    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
