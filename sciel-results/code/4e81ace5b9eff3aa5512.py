import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    best_r = 0.0
    best_pos = None
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Initializations
    def hex_init():
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        return points[:N]
        
    def random_init():
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0, 0.9, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        return np.column_stack((x, y))
        
    def ring_init():
        rings = [1, 6, 12, 18, 13]
        radii = [0, 0.3, 0.6, 0.85, 1.0]
        pts = []
        for n, r in zip(rings, radii):
            if n == 0: continue
            if n == 1:
                pts.append([0,0])
            else:
                angles = np.linspace(0, 2*np.pi, n, endpoint=False)
                xs = r * np.cos(angles)
                ys = r * np.sin(angles)
                pts.extend(zip(xs, ys))
        return np.array(pts)

    inits = [hex_init(), random_init(), random_init(), ring_init(), random_init()]
    
    def run_optimization(init_pos, mu_start=1.0):
        nonlocal best_r, best_pos
        current_time = time.time()
        if current_time - start_time > time_limit:
            return
        
        pos = init_pos.copy()
        min_pair_dist = np.min(np.linalg.norm(pos[i_idx] - pos[j_idx], axis=1))
        min_bound_dist = np.min(1 - np.linalg.norm(pos, axis=1))
        r_init = 0.5 * min(min_pair_dist, min_bound_dist) * 0.9
        
        x0 = np.concatenate([pos.flatten(), [r_init]])
        bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 0.5)]
        
        mu = mu_start
        
        def objective(v):
            if time.time() - start_time > time_limit:
                return 1e10
            pos_v = v[:2*N].reshape(N, 2)
            r_v = v[2*N]
            
            pi = pos_v[i_idx]
            pj = pos_v[j_idx]
            dists = np.linalg.norm(pi - pj, axis=1)
            dists_center = np.linalg.norm(pos_v, axis=1)
            
            v_pairs = np.maximum(0, 2*r_v - dists)
            v_bound = np.maximum(0, r_v + dists_center - 1)
            
            penalty = np.sum(v_pairs**2) + np.sum(v_bound**2)
            return -r_v + mu * penalty

        for stage in range(15):
            if time.time() - start_time > time_limit:
                break
                
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x
            pos_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N]
            
            dists_opt = np.linalg.norm(pos_opt[i_idx] - pos_opt[j_idx], axis=1)
            dists_center_opt = np.linalg.norm(pos_opt, axis=1)
            min_pair = np.min(dists_opt)
            min_bound = np.min(1 - dists_center_opt)
            
            actual_r = 0.5 * min(min_pair, min_bound)
            
            if actual_r > best_r:
                best_r = actual_r
                best_pos = pos_opt.copy()
                
            if min_pair >= 2*r_opt - 1e-5 and min_bound >= r_opt - 1e-5:
                if mu > 10000:
                    break
                mu *= 10
                x0 = x_opt
            else:
                mu *= 10
                x0 = x_opt

    for init in inits:
        if time.time() - start_time > time_limit:
            break
        run_optimization(init)
        
    if best_pos is None:
        return hex_init()
        
    return best_pos
