import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 26

def construct():
    def largest_empty_circle(centers, radii):
        if len(centers) == 0:
            return np.array([0.5, 0.5]), 0.5
        
        centers = np.array(centers)
        radii = np.array(radii)
        n = len(centers)
        
        best_r = 0
        best_c = None
        
        # 1. Voronoi Vertices
        if n >= 3:
            try:
                vor = Voronoi(centers)
                for v in vor.vertices:
                    if np.any(np.isnan(v)):
                        continue
                    if v[0] < 0 or v[0] > 1 or v[1] < 0 or v[1] > 1:
                        continue
                    r = min(v[0], 1-v[0], v[1], 1-v[1])
                    dists = np.sqrt(np.sum((centers - v)**2, axis=1))
                    r = min(r, np.min(dists - radii))
                    if r > best_r:
                        best_r = r
                        best_c = v.copy()
            except Exception:
                pass
        
        # 2. Circle-Circle Intersections
        for i in range(n):
            for j in range(i+1, n):
                c1 = centers[i]
                c2 = centers[j]
                r1 = radii[i]
                r2 = radii[j]
                d = np.linalg.norm(c1 - c2)
                if d < 1e-6 or d > r1 + r2 or d < abs(r1 - r2):
                    continue
                a = (d**2 + r1**2 - r2**2) / (2*d)
                h_sq = r1**2 - a**2
                if h_sq < 0: continue
                h = np.sqrt(h_sq)
                mid = c1 + a * (c2 - c1) / d
                perp = np.array([-(c2[1]-c1[1]), c2[0]-c1[0]]) / d
                h_vec = h * perp
                for p in [mid + h_vec, mid - h_vec]:
                    if p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1:
                        continue
                    r = min(p[0], 1-p[0], p[1], 1-p[1])
                    dists = np.sqrt(np.sum((centers - p)**2, axis=1))
                    r = min(r, np.min(dists - radii))
                    if r > best_r:
                        best_r = r
                        best_c = p.copy()

        # 3. Boundary Sampling (Robust fallback for wall-bound cases)
        # Sample points on the 4 sides
        for side in range(4):
            for t in np.linspace(0, 1, 100):
                if side == 0: p = np.array([t, 0])
                elif side == 1: p = np.array([t, 1])
                elif side == 2: p = np.array([0, t])
                else: p = np.array([1, t])
                
                r = min(p[0], 1-p[0], p[1], 1-p[1])
                dists = np.sqrt(np.sum((centers - p)**2, axis=1))
                r = min(r, np.min(dists - radii))
                if r > best_r:
                    best_r = r
                    best_c = p.copy()

        if best_c is None:
            best_c = np.array([0.5, 0.5])
            best_r = 0.01
            
        return best_c, max(best_r, 1e-6)

    def optimize_pack(centers, radii):
        n = len(centers)
        x = centers[:, 0].copy()
        y = centers[:, 1].copy()
        r = radii.copy()
        
        def objective(params):
            x_ = params[0::3]
            y_ = params[1::3]
            r_ = params[2::3]
            
            pen_bound = 0
            for i in range(n):
                pen_bound += max(0, r_[i] - x_[i])**2
                pen_bound += max(0, r_[i] - (1-x_[i]))**2
                pen_bound += max(0, r_[i] - y_[i])**2
                pen_bound += max(0, r_[i] - (1-y_[i]))**2
            
            pen_overlap = 0
            for i in range(n):
                for j in range(i+1, n):
                    d_sq = (x_[i]-x_[j])**2 + (y_[i]-y_[j])**2
                    min_d = r_[i] + r_[j]
                    if d_sq < min_d**2:
                        d = np.sqrt(d_sq)
                        pen_overlap += (min_d - d)**2
            
            return -np.sum(r_) + 50.0 * (pen_bound + pen_overlap)
        
        params0 = np.zeros(3*n)
        params0[0::3] = x
        params0[1::3] = y
        params0[2::3] = r
        
        bounds = []
        for i in range(n):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
            
        res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
        
        x_opt = res.x[0::3]
        y_opt = res.x[1::3]
        r_opt = res.x[2::3]
        
        for i in range(n):
            r_max = min(x_opt[i], 1-x_opt[i], y_opt[i], 1-y_opt[i])
            if r_opt[i] > r_max:
                r_opt[i] = r_max
        
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                min_d = r_opt[i] + r_opt[j]
                if d < min_d:
                    scale = d / min_d
                    r_opt[i] *= scale
                    r_opt[j] *= scale
                    
        return x_opt, y_opt, r_opt

    # 1. Greedy Initialization
    centers = []
    radii = []
    
    # Start with 4 corners
    centers.append([0.1, 0.1]); radii.append(0.1)
    centers.append([0.9, 0.1]); radii.append(0.1)
    centers.append([0.1, 0.9]); radii.append(0.1)
    centers.append([0.9, 0.9]); radii.append(0.1)
    
    for _ in range(N - 4):
        c, r = largest_empty_circle(centers, radii)
        centers.append(c)
        radii.append(r)
        
    centers = np.array(centers)
    radii = np.array(radii)
    
    # 2. Local Optimization
    x_opt, y_opt, r_opt = optimize_pack(centers, radii)
    
    # 3. Multi-Start
    best_score = np.sum(r_opt)
    best_x, best_y, best_r = x_opt, y_opt, r_opt
    
    np.random.seed(42)
    
    for _ in range(5):
        centers_rs = []
        radii_rs = []
        for _ in range(N):
            if len(centers_rs) == 0:
                c = np.array([0.5, 0.5]) + np.random.normal(0, 0.1, 2)
                c = np.clip(c, 0.1, 0.9)
                r = 0.1
            else:
                c, r = largest_empty_circle(centers_rs, radii_rs)
            centers_rs.append(c)
            radii_rs.append(r)
            
        centers_rs = np.array(centers_rs)
        radii_rs = np.array(radii_rs)
        
        x_rs, y_rs, r_rs = optimize_pack(centers_rs, radii_rs)
        score = np.sum(r_rs)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_rs, y_rs, r_rs

    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
