# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 11

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def project_to_triangle(p):
    if point_in_triangle(p):
        return p
    best = p
    best_dist = np.inf
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for a, b in edges:
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best = proj
    centroid = (V0 + V1 + V2) / 3.0
    best = 0.99 * best + 0.01 * centroid
    return best

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    points = x.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        # Fast barycentric check
        # p = u*V0 + v*V1 + w*V2
        # V0=(0,0), V1=(1,0), V2=(0.5, H)
        # p[1] = w*H => w = p[1]/H
        # p[0] = v + 0.5*w => v = p[0] - 0.5*w
        # u = 1 - v - w
        w = p[1] / H
        v = p[0] - 0.5 * w
        u = 1.0 - v - w
        if u < 0: penalty += u * u
        if v < 0: penalty += v * v
        if w < 0: penalty += w * w
        
    if penalty > 1e-6:
        return penalty * 1000.0
    return -min_triangle_area(points)

def generate_starts():
    starts = []
    
    # Generate a fine grid of candidates in barycentric coordinates
    num_grid = 20
    candidates = []
    for i in range(num_grid):
        for j in range(num_grid - i):
            k = num_grid - 1 - i - j
            u = i / (num_grid - 1)
            v = j / (num_grid - 1)
            w = k / (num_grid - 1)
            p = u * V0 + v * V1 + w * V2
            candidates.append(p)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    num_starts = 40
    
    for s in range(num_starts):
        np.random.seed(42 + s)
        
        # Greedy Farthest Point Sampling
        start_idx = np.random.randint(n_cand)
        selected = [candidates[start_idx].copy()]
        remaining_indices = [i for i in range(n_cand) if i != start_idx]
        
        for _ in range(N - 1):
            if not remaining_indices:
                break
            min_dists = []
            for idx in remaining_indices:
                p = candidates[idx]
                min_d = np.inf
                for sp in selected:
                    d = np.linalg.norm(p - sp)
                    if d < min_d:
                        min_d = d
                min_dists.append(min_d)
            
            max_idx_local = np.argmax(min_dists)
            max_idx_global = remaining_indices[max_idx_local]
            selected.append(candidates[max_idx_global].copy())
            remaining_indices.remove(max_idx_global)
            
        points = np.array(selected)
        
        # Deterministic jitter
        mag = 0.005
        for i in range(N):
            angle = 2 * np.pi * (s * 13 + i * 7) / 11.0
            r = mag * (0.5 + 0.5 * np.sin(s * 0.5 + i * 0.2))
            points[i] += r * np.array([np.cos(angle), np.sin(angle)])
            points[i] = project_to_triangle(points[i])
            
        starts.append(points.flatten())
        
    return starts

def construct():
    starts = generate_starts()
    best_x = None
    best_val = -np.inf
    
    for x0 in starts:
        try:
            # Stage 1: Coarse optimization
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            x_opt = res.x
            points = x_opt.reshape(-1, 2)
            
            # Stage 2: Fine polishing
            res2 = minimize(
                objective, 
                x_opt, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'adaptive': True, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            x_opt2 = res2.x
            points = x_opt2.reshape(-1, 2)
            
            # Final cleanup
            for i in range(len(points)):
                points[i] = project_to_triangle(points[i])
                
            val = min_triangle_area(points)
            if val > best_val:
                best_val = val
                best_x = points.copy()
                
        except Exception:
            continue
            
    return best_x
# EVOLVE-BLOCK-END
