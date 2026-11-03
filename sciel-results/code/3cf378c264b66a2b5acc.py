import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def min_triangle_area(points):
    min_area = np.inf
    # points is (N, 2)
    # Vectorized approach for speed?
    # Generate all pairs? No, triples.
    # 165 triples is small.
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # 2 * Area = |det(p2-p1, p3-p1)|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area

def clamp_point_to_triangle(p, v0, v1, v2):
    def dot(a, b): return np.dot(a, b)
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    
    d00 = dot(v0v1, v0v1)
    d01 = dot(v0v1, v0v2)
    d11 = dot(v0v2, v0v2)
    d02 = dot(v0v1, v0p)
    d12 = dot(v0v2, v0p)
    
    den = d00 * d11 - d01 * d01
    if den < 1e-10:
        return v0
    
    u = (d11 * d02 - d01 * d12) / den
    v = (d00 * d12 - d01 * d02) / den
    
    if u >= 0 and v >= 0 and u + v <= 1:
        return p
    
    # Project onto edges
    # Edge 0-1
    t = d02 / d00
    t = np.clip(t, 0, 1)
    proj01 = v0 + t * v0v1
    
    # Edge 1-2
    v1v2 = v2 - v1
    v1p = p - v1
    d22 = dot(v1v2, v1v2)
    d21 = dot(v1v2, v1p)
    t = d21 / d22
    t = np.clip(t, 0, 1)
    proj12 = v1 + t * v1v2
    
    # Edge 2-0
    v2v0 = v0 - v2
    v2p = p - v2
    d33 = dot(v2v0, v2v0)
    d32 = dot(v2v0, v2p)
    t = d32 / d33
    t = np.clip(t, 0, 1)
    proj20 = v2 + t * v2v0
    
    dist01 = np.linalg.norm(p - proj01)
    dist12 = np.linalg.norm(p - proj12)
    dist20 = np.linalg.norm(p - proj20)
    
    min_dist = np.min([dist01, dist12, dist20])
    if min_dist == dist01:
        return proj01
    elif min_dist == dist12:
        return proj12
    else:
        return proj20

def clamp_points(points):
    new_points = np.zeros_like(points)
    for i in range(N):
        new_points[i] = clamp_point_to_triangle(points[i], V0, V1, V2)
    return new_points

def objective(x):
    # x is flat array of 2*N
    points = x.reshape(N, 2)
    # Optional: clamp here to keep optimizer in bounds?
    # Nelder-Mead doesn't handle bounds well.
    # But if we clamp, the function becomes non-smooth at the boundary.
    # It's often better to let it go outside and penalize, or just clamp and accept the kink.
    # Clamping is safer for validity.
    points = clamp_points(points)
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Symmetric initial guess
    # Center
    center = np.array([0.5, np.sqrt(3)/6])
    # Hexagon
    r = 0.2
    hex_pts = []
    for i in range(6):
        angle = i * np.pi / 3
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        hex_pts.append([x, y])
    # 4 more points?
    # Let's just use random restarts primarily.
    
    num_restarts = 0
    while time.time() < deadline:
        # Generate random points inside triangle
        # Use barycentric coords
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        
        points = np.zeros((N, 2))
        points[:, 0] = V0[0] + u * (V1[0] - V0[0]) + v * (V2[0] - V0[0])
        points[:, 1] = V0[1] + u * (V1[1] - V0[0]) + v * (V2[1] - V0[1]) # Wait, V1[1] is 0
        
        # Correct barycentric to cartesian:
        # P = (1-u-v)*V0 + u*V1 + v*V2
        # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
        # P_x = u * 1 + v * 0.5
        # P_y = v * sqrt(3)/2
        
        points[:, 0] = u + 0.5 * v
        points[:, 1] = np.sqrt(3)/2 * v
        
        x0 = points.flatten()
        
        # Nelder-Mead
        # maxiter depends on remaining time?
        # Let's fix it to 200 iterations per restart.
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
            x_opt = res.x
            points_opt = x_opt.reshape(N, 2)
            points_opt = clamp_points(points_opt)
            score = min_triangle_area(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
        except Exception:
            pass
        
        num_restarts += 1
        
        # If we have time, do a local nudge on the best point
        if best_points is not None and time.time() < deadline - 5.0:
            # Nudge logic
            # Find min area triangle
            min_area = np.inf
            min_idx = (0,0,0)
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        p1, p2, p3 = best_points[i], best_points[j], best_points[k]
                        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                        if area < min_area:
                            min_area = area
                            min_idx = (i, j, k)
            
            i, j, k = min_idx
            centroid = (best_points[i] + best_points[j] + best_points[k]) / 3.0
            
            # Try nudging
            for eps in [0.001, 0.005, 0.01, 0.02]:
                new_points = best_points.copy()
                # Move i, j, k away from centroid
                for idx in [i, j, k]:
                    vec = new_points[idx] - centroid
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        new_points[idx] += eps * vec / norm
                
                new_points = clamp_points(new_points)
                new_score = min_triangle_area(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points

    return best_points
