import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit
    
    # 1. Initialization: Grid + Jitter
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5, 0.5, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    zs = np.linspace(-0.5, 0.5, nz)
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    points = np.array(points[:N])
    
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    best_points = points.copy()
    
    def compute_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        wall_dists = np.min(np.stack([
            0.5 - np.abs(pts[:, 0]),
            0.5 - np.abs(pts[:, 1]),
            0.5 - np.abs(pts[:, 2])
        ], axis=1), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    best_radius = compute_radius(best_points)
    
    # 2. Simulated Annealing
    T = 0.05
    T_min = 0.0001
    cooling_rate = 0.999
    
    current_points = best_points.copy()
    current_radius = best_radius
    
    steps = 0
    while time.time() - start_time < 60.0:
        step_size = 0.01 * (T / 0.05)
        new_points = current_points + np.random.normal(0, step_size, current_points.shape)
        
        new_radius = compute_radius(new_points)
        
        delta = new_radius - current_radius
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_points = new_points
            current_radius = new_radius
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
        steps += 1
        
        if steps % 1000 == 0 and time.time() - start_time > 55.0:
            break

    # 3. Polish with Binary Search + L-BFGS-B
    def energy_for_radius(pts, r):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        pair_violations = np.maximum(0, 2 * r - dists)
        pair_energy = np.sum(pair_violations ** 2)
        
        wall_dists = np.min(np.stack([
            0.5 - np.abs(pts[:, 0]),
            0.5 - np.abs(pts[:, 1]),
            0.5 - np.abs(pts[:, 2])
        ], axis=1), axis=1)
        
        wall_violations = np.maximum(0, r - wall_dists)
        wall_energy = np.sum(wall_violations ** 2)
        
        return pair_energy + wall_energy

    def flatten_points(pts):
        return pts.flatten()

    def unflatten_points(vec):
        return vec.reshape(N, 3)

    r_low = best_radius * 0.98
    r_high = best_radius * 1.02
    
    best_final_points = best_points.copy()
    best_final_radius = best_radius
    
    while time.time() - start_time < time_limit:
        r_mid = (r_low + r_high) / 2
        
        x0 = flatten_points(best_final_points)
        
        def obj(x):
            return energy_for_radius(unflatten_points(x), r_mid)
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        except Exception:
            res = None
        
        if res is not None and res.fun < 1e-8:
            r_low = r_mid
            best_final_points = unflatten_points(res.x)
            best_final_radius = r_mid
        else:
            r_high = r_mid
            
        if r_high - r_low < 1e-6:
            break

    # Final safety clip
    best_final_points = np.clip(best_final_points, -0.5, 0.5)
    
    return best_final_points
