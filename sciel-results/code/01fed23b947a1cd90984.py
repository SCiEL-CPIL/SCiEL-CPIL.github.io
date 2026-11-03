import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 110.0  # 10s margin
    
    def optimize(points, r, max_iters=500):
        points = points.copy()
        for it in range(max_iters):
            if time.time() - start_time > timeout:
                return False, points
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            
            force = np.zeros_like(points)
            
            mask = dist < 2 * r
            dist_safe = np.where(dist > 1e-10, dist, 1.0)
            
            push_mag = np.where(mask, (2 * r - dist_safe) / 2.0, 0.0)
            
            unit_vec = -diff / dist_safe[:, :, np.newaxis]
            force_contrib = unit_vec * push_mag[:, :, np.newaxis]
            
            force += np.sum(force_contrib, axis=1)
            
            dist_to_bound = 1.0 - np.linalg.norm(points, axis=1)
            mask_bound = dist_to_bound < r
            
            if np.any(mask_bound):
                norms = np.linalg.norm(points, axis=1)
                norms_safe = np.where(norms > 1e-10, norms, 1.0)
                
                b_push_mag = np.where(mask_bound, r - dist_to_bound, 0.0)
                b_dir = -points / norms_safe[:, np.newaxis]
                
                force += b_dir * b_push_mag[:, np.newaxis]
            
            points += force * 0.5
            
            # Check convergence
            dist_new = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
            np.fill_diagonal(dist_new, np.inf)
            min_dist = np.min(dist_new)
            max_violation_internal = max(0, 2 * r - min_dist)
            
            dist_to_bound_new = 1.0 - np.linalg.norm(points, axis=1)
            min_dist_bound = np.min(dist_to_bound_new)
            max_violation_bound = max(0, r - min_dist_bound)
            
            max_violation = max(max_violation_internal, max_violation_bound)
            
            if max_violation < 1e-8:
                return True, points
            
        return False, points

    def generate_initial_config(seed):
        rng = np.random.RandomState(seed)
        spacing = 0.25
        k = 10
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= (1 - 0.1)**2:
                    rows.append([x, y])
        
        points = np.array(rows)
        if len(points) < N:
            while len(points) < N:
                p = rng.uniform(-0.8, 0.8, 2)
                if np.linalg.norm(p) < 0.9:
                    points = np.vstack([points, p])
        
        rng.shuffle(points)
        points = points[:N]
        points += rng.normal(0, 0.01, points.shape)
        
        return points

    best_r = 0.0
    best_points = None
    
    # Try multiple seeds
    for seed in range(20):
        if time.time() - start_time > timeout:
            break
            
        points = generate_initial_config(seed)
        
        lo, hi = 0.1, 0.13
        
        best_r_local = lo
        best_points_local = points.copy()
        
        for _ in range(25):
            if time.time() - start_time > timeout:
                break
                
            mid = (lo + hi) / 2
            success, opt_points = optimize(points, mid)
            if success:
                best_r_local = mid
                best_points_local = opt_points
                lo = mid
            else:
                hi = mid
        
        if best_r_local > best_r:
            best_r = best_r_local
            best_points = best_points_local
            
    if best_points is None:
        # Fallback to seed program logic if optimization fails completely
        # But this shouldn't happen
        return generate_initial_config(0)
        
    return best_points
