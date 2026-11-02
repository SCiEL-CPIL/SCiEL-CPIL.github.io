import numpy as np
from scipy.optimize import minimize
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def project_to_triangle(p):
    if point_in_triangle(p):
        return p
    
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    min_dist = float('inf')
    best_p = p
    
    for a, b in edges:
        ab = b - a
        ap = p - a
        if np.dot(ab, ab) == 0:
            proj = a
        else:
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
        
        dist = np.linalg.norm(p - proj)
        if dist < min_dist:
            min_dist = dist
            best_p = proj
            
    return best_p

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective(points_flat):
    points = points_flat.reshape(-1, 2)
    
    # Penalty for points outside
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            # Calculate distance to boundary
            # Re-using projection logic is expensive, so approximate or just use a fixed penalty
            # Fixed penalty is simpler and often sufficient
            penalty += 100.0 * (1.0 / (1.0 + np.linalg.norm(p))) # Just a dummy, let's use fixed
            penalty = 100.0 # Flat penalty
    
    if penalty > 0:
        return -min_triangle_area(points) + penalty
    
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Random Starts
    for _ in range(30):
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    p = u * V0 + v * V1 + (1 - u - v) * V2
                    pts.append(p)
                    break
        candidates.append(np.array(pts))
        
    # 2. Grid Start
    # Generate grid
    grid_u = np.linspace(0.05, 0.95, 10)
    grid_v = np.linspace(0.05, 0.95, 10)
    all_grid_pts = []
    for u in grid_u:
        for v in grid_v:
            w = 1 - u - v
            if w >= 0.05:
                p = u * V0 + v * V1 + w * V2
                all_grid_pts.append(p)
                
    if len(all_grid_pts) >= N:
        # Greedy selection
        selected = [all_grid_pts[0]]
        remaining = all_grid_pts[1:]
        while len(selected) < N and remaining:
            best_pt = None
            best_min_dist = -1
            for p in remaining:
                min_d = min(np.linalg.norm(p - s) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pt = p
            if best_pt:
                selected.append(best_pt)
                remaining.remove(best_pt)
        if len(selected) == N:
            candidates.append(np.array(selected))

    # Optimization
    for start_pts in candidates:
        if time.time() > deadline:
            break
            
        x0 = start_pts.flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            
            opt_pts = res.x.reshape(-1, 2)
            # Project to ensure validity
            opt_pts = np.array([project_to_triangle(p) for p in opt_pts])
            
            score = min_triangle_area(opt_pts)
            
            if score > best_score:
                best_score = score
                best_points = opt_pts
                
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0]
        
    return best_points
