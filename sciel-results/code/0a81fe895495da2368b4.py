"""Improved program for the suite task; uses multi-start smooth optimization with hexagonal lattice inits."""
import numpy as np
from scipy.optimize import minimize

N = 50


def generate_hex_init(row_counts):
    """Generate initial points based on a hexagonal grid with specified row counts."""
    points = []
    s = 0.15
    y_spacing = s * np.sqrt(3) / 2
    
    total_height = y_spacing * (len(row_counts) - 1)
    start_y = -total_height / 2.0
    
    for i, count in enumerate(row_counts):
        y = start_y + i * y_spacing
        offset = (s / 2.0) if i % 2 != 0 else 0.0
        
        total_width = s * (count - 1)
        start_x = -total_width / 2.0 + offset
        
        for j in range(count):
            x = start_x + j * s
            points.append([x, y])
            
    pts = np.array(points)
    return pts

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(params, k):
        centers = params.reshape(N, 2)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        pair_terms = dists / 2.0
        wall_dists = 0.5 - np.abs(centers)
        
        all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
        all_terms = all_terms[np.isfinite(all_terms)]
        
        if len(all_terms) == 0:
            return 0.0
            
        m = np.min(all_terms)
        shifted = all_terms - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    def evaluate(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        return min(min_pair, min_wall)

    best_centers = None
    best_score = 0.0
    
    configs = [
        [1, 8, 8, 8, 8, 8, 8, 1], 
        [5, 8, 8, 8, 8, 8, 5],    
        [10, 8, 8, 8, 8, 6, 2],   
    ]
    
    valid_configs = []
    for cfg in configs:
        if sum(cfg) == N:
            valid_configs.append(cfg)
            
    if not valid_configs:
        valid_configs = [[1, 8, 8, 8, 8, 8, 8, 1]]

    np.random.seed(123)
    random_init = np.random.uniform(-0.45, 0.45, (N, 2))
    
    for cfg in valid_configs:
        init_pts = generate_hex_init(cfg)
        
        if len(init_pts) > N:
            init_pts = init_pts[:N]
        elif len(init_pts) < N:
            extra = np.random.uniform(-0.45, 0.45, (N - len(init_pts), 2))
            init_pts = np.vstack([init_pts, extra])
            
        init_pts = init_pts + np.random.uniform(-0.01, 0.01, init_pts.shape)
        
        try:
            result = minimize(
                objective, 
                init_pts.reshape(-1), 
                args=(50,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            centers = result.x.reshape(N, 2)
            
            result2 = minimize(
                objective, 
                centers.reshape(-1), 
                args=(200,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            centers = result2.x.reshape(N, 2)
            
            score = evaluate(centers)
            
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
        except Exception:
            pass
            
    try:
        result = minimize(
            objective, 
            random_init.reshape(-1), 
            args=(50,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        centers = result.x.reshape(N, 2)
        result2 = minimize(
            objective, 
            centers.reshape(-1), 
            args=(200,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        centers = result2.x.reshape(N, 2)
        score = evaluate(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    except Exception:
        pass
        
    if best_centers is None:
        best_centers = random_init
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
