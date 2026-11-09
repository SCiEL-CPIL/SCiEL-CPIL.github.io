import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def hex_lattice_init():
        best_points = None
        best_count_diff = 1000
        for scale in np.linspace(0.04, 0.08, 20):
            points = []
            j_max = int(1.0 / (scale * np.sqrt(3) / 2)) + 1
            i_max = int(1.0 / scale) + 1
            for j in range(-j_max, j_max + 1):
                y = j * scale * np.sqrt(3) / 2
                if abs(y) > 1.0:
                    continue
                for i in range(-i_max, i_max + 1):
                    x = i * scale + (j % 2) * scale / 2
                    if x*x + y*y <= 1.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) == 0:
                continue
            diff = abs(len(points) - N)
            if diff < best_count_diff:
                best_count_diff = diff
                best_points = points.copy()
            if len(points) == N:
                break
        if best_points is None:
            return np.random.randn(N, 2) * 0.5
        if len(best_points) > N:
            dists = np.linalg.norm(best_points, axis=1)
            idx = np.argsort(dists)
            best_points = best_points[idx[:N]]
        elif len(best_points) < N:
            # Pad with random
            pass
        best_points += np.random.randn(*best_points.shape) * 0.005
        return best_points

    def soft_min(values, beta):
        m = np.min(values)
        exps = np.exp(-beta * (values - m))
        return m - np.log(np.sum(exps)) / beta

    def objective(x):
        points = x.reshape(-1, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        radii = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii
        
        vals = np.concatenate([pair_dists.ravel(), boundary_dists])
        vals = vals[vals < np.inf]
        
        beta = 10.0
        return -soft_min(vals, beta)

    best_x = None
    best_score = -np.inf
    
    num_restarts = 3 # Keep it low to ensure time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        x0 = hex_lattice_init()
        
        # Use numerical gradient
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success:
            x_opt = res.x
            points = x_opt.reshape(-1, 2)
            if not np.all(np.isfinite(points)):
                continue
            
            dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            radii = np.linalg.norm(points, axis=1)
            min_bound = np.min(1.0 - radii)
            
            score = min(min_pair, min_bound)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
    if best_x is None:
        return hex_lattice_init()
        
    return best_x
