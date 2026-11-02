import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13
# Precompute triangle indices
tri_i = []
tri_j = []
tri_k = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            tri_i.append(i)
            tri_j.append(j)
            tri_k.append(k)
tri_i = np.array(tri_i, dtype=np.int32)
tri_j = np.array(tri_j, dtype=np.int32)
tri_k = np.array(tri_k, dtype=np.int32)

def construct():
    start_time = time.time()
    time_limit = 145.0 # 5s margin

    def compute_score(points):
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0
        if area_hull < 1e-9:
            return 0.0, 0.0
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, area_hull

    def soft_min_objective(points_flat, beta):
        points = points_flat.reshape(-1, 2)
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
        if area_hull < 1e-9:
            return 1e6
        
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area / area_hull

    def hard_objective(points_flat):
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def regular_polygon(n, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    best_points = regular_polygon(N)
    best_score, _ = compute_score(best_points)

    # Multi-start optimization
    # 1. Regular polygon
    # 2. Regular polygon with small angular noise
    # 3. Regular polygon with radial noise
    
    rng = np.random.RandomState(42)
    
    starts = []
    # Start 1: Perfect regular
    starts.append(regular_polygon(N))
    
    # Start 2-5: Angular perturbations
    for i in range(4):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.randn(N) * 0.05
        radii = np.ones(N) + rng.randn(N) * 0.05
        starts.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))
    
    # Start 6-10: Larger perturbations
    for i in range(5):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.randn(N) * 0.2
        radii = np.ones(N) + rng.randn(N) * 0.1
        starts.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    for start_pts in starts:
        if time.time() > start_time + time_limit - 10:
            break
        
        current = start_pts.copy()
        
        # Annealing with Soft-Min
        betas = [10, 50, 200, 1000]
        for beta in betas:
            if time.time() > start_time + time_limit - 10:
                break
            try:
                res = opt.minimize(
                    lambda x: soft_min_objective(x, beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Polish with Hard Objective (Nelder-Mead)
        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    hard_objective,
                    current.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                candidate = res.x.reshape(-1, 2)
                s, _ = compute_score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate.copy()
            except Exception:
                pass

    return best_points
