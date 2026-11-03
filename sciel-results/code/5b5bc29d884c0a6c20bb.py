"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def compute_radii(centers):
        """
        Vectorized computation of maximum non-overlapping radii for given centers.
        centers: (N, 2) array
        returns: (N,) array of radii
        """
        centers = np.asarray(centers)
        # Boundary distances
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        boundary_r = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        
        # For each i, the max radius is min(boundary_r[i], min_{j != i} dist[i, j] / 2)
        np.fill_diagonal(dist, np.inf)
        
        # min distance to any other center, divided by 2
        min_neighbor_dist = np.min(dist, axis=1) / 2.0
        
        radii = np.minimum(boundary_r, min_neighbor_dist)
        return radii

    def objective(centers):
        """
        Returns negative sum of radii (to be minimized).
        """
        radii = compute_radii(centers)
        return -np.sum(radii)

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        # Check bounds
        if np.any(xs - rs < -tol) or np.any(xs + rs > 1 + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > 1 + tol):
            return False
        
        # Check overlaps
        centers = x[:, :2]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        
        # sum of radii
        r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        # Check if dist < r_sum - tol
        mask = ~np.eye(N, dtype=bool)
        if np.any(dist[mask] < r_sum[mask] - tol):
            return False
            
        return True

    def fix_validity(x):
        """
        If x is invalid, shrink radii until valid.
        """
        if is_valid(x, tol=1e-8):
            return x
        
        # Try shrinking radii
        factor = 1.0
        for _ in range(100):
            factor *= 0.99
            new_x = x.copy()
            new_x[:, 2] = x[:, 2] * factor
            if is_valid(new_x, tol=1e-8):
                return new_x
        
        # Fallback: very small radii
        new_x = x.copy()
        new_x[:, 2] = 0.01
        return new_x

    def simulated_annealing_vectorized(init_centers, max_iter=20000, t_start=0.1, t_end=0.001):
        """
        Fast SA with vectorized objective.
        """
        current = init_centers.copy()
        current_obj = objective(current)
        best = current.copy()
        best_obj = current_obj
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iter)
        
        rng = np.random.RandomState(42)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Propose move: pick random circle, perturb
            i = rng.randint(N)
            # Scale step size with temperature
            step_scale = 0.05 * (t / t_start) + 0.001
            dx = rng.normal(0, step_scale)
            dy = rng.normal(0, step_scale)
            
            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.0, 1.0)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.0, 1.0)
            
            new_obj = objective(new_centers)
            delta = new_obj - current_obj
            
            # Accept if better or with prob exp(-delta/t)
            if delta < 0 or rng.random() < np.exp(-delta / t):
                current = new_centers
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best = current.copy()
                    best_obj = current_obj
            
            t *= cooling
        
        return best, best_obj

    def local_search_vectorized(centers, max_iter=5000):
        """
        Discrete local search, vectorized objective.
        """
        current = centers.copy()
        current_obj = objective(current)
        step_scale = 0.02
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try each circle
            for i in range(N):
                # Try 8 directions
                for dx in [-step_scale, 0, step_scale]:
                    for dy in [-step_scale, 0, step_scale]:
                        if dx == 0 and dy == 0:
                            continue
                        new_centers = current.copy()
                        new_x = np.clip(current[i, 0] + dx, 0.0, 1.0)
                        new_y = np.clip(current[i, 1] + dy, 0.0, 1.0)
                        new_centers[i, 0] = new_x
                        new_centers[i, 1] = new_y
                        
                        new_obj = objective(new_centers)
                        if new_obj < current_obj - 1e-10:
                            current = new_centers
                            current_obj = new_obj
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                step_scale *= 0.8
                if step_scale < 0.001:
                    break
        
        return current, current_obj

    # Initializations
    inits = []
    # Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_centers = []
    for i in range(side):
        for j in range(side):
            if len(grid_centers) < N:
                grid_centers.append([(i + 0.5) / side, (j + 0.5) / side])
    inits.append(np.array(grid_centers[:N]))
    
    # Randoms
    for seed in [1, 7, 13, 23, 37, 53, 71, 89, 101, 113]:
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(0.1, 0.9, size=(N, 2)))

    best_result = None
    best_score = np.inf

    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Phase 1: SA
        sa_centers, sa_obj = simulated_annealing_vectorized(init_centers, max_iter=10000)
        
        # Phase 2: Local Search
        ls_centers, ls_obj = local_search_vectorized(sa_centers, max_iter=2000)
        
        # Compute radii
        radii = compute_radii(ls_centers)
        x_full = np.column_stack([ls_centers, radii])
        
        # Ensure validity
        x_valid = fix_validity(x_full)
        
        score = -np.sum(x_valid[:, 2])
        if score < best_score:
            best_score = score
            best_result = x_valid.copy()

    # Fallback
    if best_result is None:
        centers = inits[0]
        radii = compute_radii(centers)
        best_result = np.column_stack([centers, radii])
        best_result = fix_validity(best_result)

    return best_result

# EVOLVE-BLOCK-END
