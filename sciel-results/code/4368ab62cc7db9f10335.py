import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    def unpack(params):
        radii = params[:n]
        angle_deltas = params[n:]
        radii = np.maximum(radii, 0.1)
        angles = base_angles + angle_deltas
        sort_idx = np.argsort(angles)
        angles = angles[sort_idx]
        radii = radii[sort_idx]
        return radii, angles

    def calc_score(params):
        radii, angles = unpack(params)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        dists_to_centroid = np.linalg.norm(pts, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 0.0
        pts = pts / mean_dist
        
        cross_products = np.zeros(n)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i+1)%n]
            p2 = pts[(i+2)%n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross_products[i] = v1[0]*v2[1] - v1[1]*v2[0]
            
        if np.any(cross_products == 0) or not (np.all(cross_products > 0) or np.all(cross_products < 0)):
            return 0.0
            
        min_vertex_score = 1.0
        for i in range(n):
            v = pts[i]
            dists = np.linalg.norm(pts - v, axis=1)
            dists = np.delete(dists, i)
            dists = np.sort(dists)
            
            if dists[0] < 0.05:
                return 0.0
                
            diffs = np.diff(dists)
            k = np.argmin(diffs)
            D = (dists[k] + dists[k+1]) / 2.0
            if D < 0.1:
                return 0.0
                
            if k == 0:
                sel = dists[0:4]
            elif k == 17:
                sel = dists[15:19]
            else:
                sel = dists[k-1:k+3]
                
            ratios = np.maximum(sel / D, D / sel)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)
            min_vertex_score = min(min_vertex_score, score)
            
        return min_vertex_score

    def objective(params):
        score = calc_score(params)
        if score == 0.0:
            return 1.0
        return -score

    init_radii = np.ones(n)
    init_deltas = np.zeros(n)
    x0 = np.concatenate([init_radii, init_deltas])
    
    best_score = calc_score(x0)
    best_params = x0.copy()
    
    np.random.seed(42)
    
    for i in range(10):
        radii = np.random.uniform(0.8, 1.2, n)
        deltas = np.random.uniform(-0.05, 0.05, n)
        x_start = np.concatenate([radii, deltas])
        
        res = minimize(objective, x_start, method='Powell', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        
        s = calc_score(res.x)
        if s > best_score:
            best_score = s
            best_params = res.x.copy()
            
    radii, angles = unpack(best_params)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0.01:
        pts = pts / mean_dist
        
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0]*v2[1] - v1[1]*v2[0]
        
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)) or np.any(np.abs(cross_products) < 1e-6):
        angles_reg = np.linspace(0, 2*np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
        
    return pts
