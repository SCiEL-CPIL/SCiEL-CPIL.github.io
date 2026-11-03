import numpy as np
from scipy.optimize import minimize
import time

N = 11

def get_unit_hex_vertices():
    return np.array([
        [0.0, 1.0],
        [np.sqrt(3)/2, 0.5],
        [np.sqrt(3)/2, -0.5],
        [0.0, -1.0],
        [-np.sqrt(3)/2, -0.5],
        [-np.sqrt(3)/2, 0.5]
    ])

UNIT_VERTS = get_unit_hex_vertices()

# Normals for pointy-top outer hexagon
NORMALS = np.array([[np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)])

def get_hex_vertices(center, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return center + UNIT_VERTS @ R.T

def point_in_poly(point, poly):
    sign = 0
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1)%n]
        edge = p2 - p1
        vec = point - p1
        cross = edge[0]*vec[1] - edge[1]*vec[0]
        if cross != 0:
            if sign == 0:
                sign = 1 if cross > 0 else -1
            else:
                if (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                    return False
    return True

def min_dist_poly(poly1, poly2):
    # Bounding box check
    bb1_min = poly1.min(axis=0)
    bb1_max = poly1.max(axis=0)
    bb2_min = poly2.min(axis=0)
    bb2_max = poly2.max(axis=0)
    
    if bb1_max[0] < bb2_min[0] or bb1_max[1] < bb2_min[1] or bb2_max[0] < bb1_min[0] or bb2_max[1] < bb1_min[1]:
        # Disjoint boxes, min distance is box distance
        dx = max(0, bb2_min[0] - bb1_max[0], bb1_min[0] - bb2_max[0])
        dy = max(0, bb2_min[1] - bb1_max[1], bb1_min[1] - bb2_max[1])
        return np.sqrt(dx*dx + dy*dy)
        
    # Check for overlap
    for p in poly1:
        if point_in_poly(p, poly2):
            return 0.0
    for p in poly2:
        if point_in_poly(p, poly1):
            return 0.0
    
    # Min distance
    min_d = float('inf')
    for p in poly1:
        for i in range(len(poly2)):
            p1 = poly2[i]
            p2 = poly2[(i+1)%len(poly2)]
            v = p2 - p1
            w = p - p1
            c1 = np.dot(w, v)
            if c1 <= 0:
                d = np.linalg.norm(p - p1)
            else:
                c2 = np.dot(w, -v)
                if c2 <= 0:
                    d = np.linalg.norm(p - p2)
                else:
                    b = np.dot(v, v)
                    t = c1 / b
                    proj = p1 + t * v
                    d = np.linalg.norm(p - proj)
            if d < min_d:
                min_d = d
    return min_d

def h_unit(w):
    return np.max(UNIT_VERTS @ w)

def objective(params):
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i] = params[3*i:3*i+2]
        angles[i] = params[3*i+2]
    
    S_min = 0.0
    for i in range(N):
        angle_rad = np.deg2rad(angles[i])
        R_i = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        for k in range(6):
            n_k = NORMALS[k]
            w = R_i.T @ n_k
            h_val = h_unit(w)
            c_dot_n = np.dot(centers[i], n_k)
            S_req = (c_dot_n + h_val) / (np.sqrt(3)/2)
            if S_req > S_min:
                S_min = S_req
                
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d_center = np.linalg.norm(centers[i] - centers[j])
            if d_center < 2.0:
                v1 = get_hex_vertices(centers[i], angles[i])
                v2 = get_hex_vertices(centers[j], angles[j])
                d = min_dist_poly(v1, v2)
                if d < 1e-6:
                    penalty += (1e-6 - d)**2
    
    return S_min + 100.0 * penalty

def construct():
    sqrt3 = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0.0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5),
        (sqrt3, 1.5),
        (sqrt3/2, 3.0),
        (-sqrt3/2, 3.0),
        (-sqrt3, 1.5)
    ]
    angles = [0.0] * N
    
    params0 = []
    for i in range(N):
        params0.append(centers[i][0])
        params0.append(centers[i][1])
        params0.append(angles[i])
    
    start_time = time.time()
    timeout = 140.0
    
    res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 2000, 'maxfun': 20000, 'max_time': timeout})
    
    best_params = res.x
    
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
        
    # Calculate final S
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i] = [best_params[3*i], best_params[3*i+1]]
        angles[i] = best_params[3*i+2]
        
    S_min = 0.0
    for i in range(N):
        angle_rad = np.deg2rad(angles[i])
        R_i = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        for k in range(6):
            n_k = NORMALS[k]
            w = R_i.T @ n_k
            h_val = h_unit(w)
            c_dot_n = np.dot(centers[i], n_k)
            S_req = (c_dot_n + h_val) / (np.sqrt(3)/2)
            if S_req > S_min:
                S_min = S_req
                
    return inner, (0.0, 0.0), S_min, 0.0
