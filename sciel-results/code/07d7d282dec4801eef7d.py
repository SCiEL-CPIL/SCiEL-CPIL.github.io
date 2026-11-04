import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    def compute_radii(positions):
        xs = positions[:N]
        ys = positions[N:]
        
        # Wall constraints
        wall_rs = np.minimum.reduce([xs, 1 - xs, ys, 1 - ys])
        
        # Neighbor constraints
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        dist_mat = np.sqrt(dx*dx + dy*dy)
        
        # Set diagonal to inf so it doesn't affect min
        np.fill_diagonal(dist_mat, np.inf)
        
        # Min distance to any other circle
        min_dists = np.min(dist_mat, axis=1)
        
        # Heuristic: r_i <= min_dists[i] / 2
        neighbor_rs = min_dists / 2.0
        
        rs = np.minimum(wall_rs, neighbor_rs)
        return rs

    def objective(positions):
        rs = compute_radii(positions)
        return -np.sum(rs)

    def repulsion_init():
        pos = np.random.rand(2*N)
        pos = np.clip(pos, 0.05, 0.95)
        for _ in range(50):
            forces = np.zeros(2*N)
            for i in range(N):
                xi, yi = pos[i], pos[N+i]
                fx, fy = 0, 0
                for j in range(N):
                    if i == j: continue
                    xj, yj = pos[j], pos[N+j]
                    dx = xi - xj
                    dy = yi - yj
                    d2 = dx*dx + dy*dy
                    if d2 < 1e-6:
                        dx = np.random.normal(0, 0.01)
                        dy = np.random.normal(0, 0.01)
                        d2 = dx*dx + dy*dy
                    d = np.sqrt(d2)
                    f = 1.0 / (d2 * d)
                    fx += f * dx
                    fy += f * dy
                if xi < 0.1: fx += 5
                if xi > 0.9: fx -= 5
                if yi < 0.1: fy += 5
                if yi > 0.9: fy -= 5
                
                forces[i] += fx
                forces[N+i] += fy
            
            pos += forces * 0.005
            pos = np.clip(pos, 0.01, 0.99)
        return pos

    best_pos = None
    best_score = -np.inf
    
    import time
    start_time = time.time()
    time_limit = 110
    
    restarts = 0
    while time.time() - start_time < time_limit:
        restarts += 1
        current_pos = repulsion_init()
        current_obj = objective(current_pos)
        
        T = 0.1
        T_min = 0.0001
        alpha = 0.995
        steps = 1000
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            new_pos = current_pos.copy()
            idx = np.random.randint(2*N)
            delta = np.random.normal(0, T)
            new_pos[idx] += delta
            new_pos[idx] = np.clip(new_pos[idx], 0.0, 1.0)
            
            new_obj = objective(new_pos)
            delta_obj = new_obj - current_obj
            
            if delta_obj < 0 or np.random.rand() < np.exp(-delta_obj / T):
                current_pos = new_pos
                current_obj = new_obj
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        # Local Refinement
        try:
            res = minimize(objective, current_pos, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < current_obj:
                current_pos = res.x
                current_obj = res.fun
        except:
            pass

        rs = compute_radii(current_pos)
        score = np.sum(rs)
        
        if score > best_score:
            best_score = score
            best_pos = current_pos.copy()

    xs = best_pos[:N]
    ys = best_pos[N:]
    rs = compute_radii(best_pos)
    
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = rs
    
    return result
