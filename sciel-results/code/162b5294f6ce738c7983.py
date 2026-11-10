import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety (150s total)
    
    best_r = 0.0
    best_centers = None
    
    def get_min_radius(centers):
        """Calculate the exact maximum radius for a given configuration."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist / 2.0, min_wall_dist)

    def compute_gradients(centers, beta):
        """
        Compute the gradient of the soft-min objective with respect to centers.
        """
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e10)
        
        wall_dists = np.empty((N, 6))
        wall_dists[:, 0] = centers[:, 0] + 0.5
        wall_dists[:, 1] = 0.5 - centers[:, 0]
        wall_dists[:, 2] = centers[:, 1] + 0.5
        wall_dists[:, 3] = 0.5 - centers[:, 1]
        wall_dists[:, 4] = centers[:, 2] + 0.5
        wall_dists[:, 5] = 0.5 - centers[:, 2]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dist[mask]
        wall_flat = 2.0 * wall_dists.flatten()
        
        all_d = np.concatenate([pair_dists, wall_flat])
        max_d = np.max(all_d)
        
        exp_pairs = np.exp(-beta * (pair_dists - max_d))
        exp_walls = np.exp(-beta * (wall_flat - max_d))
        
        sum_pairs = np.sum(exp_pairs)
        sum_walls = np.sum(exp_walls)
        total = sum_pairs + sum_walls
        
        if total < 1e-300:
            return np.zeros_like(centers)
        
        w_pairs = exp_pairs / total
        w_walls = exp_walls / total
        
        grad = np.zeros_like(centers)
        
        W_pair_mat = np.zeros((N, N))
        W_pair_mat[mask] = w_pairs
        
        U_pair = np.zeros((N, N))
        U_pair[mask] = w_pairs / (pair_dists + 1e-10)
        
        row_sums_U = np.sum(U_pair, axis=1)
        weighted_c = U_pair @ centers
        
        grad_pairs = centers * row_sums_U[:, np.newaxis] - weighted_c
        
        W_wall_mat = w_walls.reshape(N, 6)
        
        grad_walls = np.zeros_like(centers)
        grad_walls[:, 0] = W_wall_mat[:, 0] - W_wall_mat[:, 1]
        grad_walls[:, 1] = W_wall_mat[:, 2] - W_wall_mat[:, 3]
        grad_walls[:, 2] = W_wall_mat[:, 4] - W_wall_mat[:, 5]
        
        grad_soft_min = grad_pairs + grad_walls
        return -grad_soft_min

    def soft_min_value(centers, beta):
        """Compute the soft-min value (for monitoring)."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e10)
        
        wall_dists = 0.5 - np.abs(centers)
        d_pair = dist[np.triu_indices(N, k=1)]
        d_wall = 2.0 * wall_dists.flatten()
        
        all_d = np.concatenate([d_pair, d_wall])
        max_d = np.max(all_d)
        log_sum = np.log(np.sum(np.exp(-beta * (all_d - max_d))))
        soft_min = -log_sum / beta + max_d
        
        return soft_min / 2.0

    def force_directed_relax(centers, max_steps=15000, initial_step=0.003, repulsion_strength=1.0):
        """
        Force-directed relaxation with soft repulsion from all neighbors.
        Combines global repulsion with adaptive step size.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = initial_step
        no_improve_count = 0
        max_no_improve = 500
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Compute repulsion forces
            inv_dist_sq = 1.0 / (dist**2 + 1e-10)
            inv_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, inv_dist_sq)
            inv_dist_sq = np.minimum(inv_dist_sq, 50.0)
            
            row_sums = np.sum(inv_dist_sq, axis=1)
            weighted_centers = inv_dist_sq @ centers
            
            forces = centers * row_sums[:, np.newaxis] - weighted_centers
            forces *= repulsion_strength
            
            # Wall repulsion
            for k in range(3):
                dist_to_pos_wall = 0.5 - centers[:, k]
                dist_to_neg_wall = 0.5 + centers[:, k]
                
                wall_force = np.zeros(N)
                
                close_to_pos = dist_to_pos_wall < 0.15
                if np.any(close_to_pos):
                    wall_force[close_to_pos] += repulsion_strength * 0.5 / (dist_to_pos_wall[close_to_pos]**2 + 1e-10)
                    wall_force[close_to_pos] = np.minimum(wall_force[close_to_pos], 50.0)
                
                close_to_neg = dist_to_neg_wall < 0.15
                if np.any(close_to_neg):
                    wall_force[close_to_neg] -= repulsion_strength * 0.5 / (dist_to_neg_wall[close_to_neg]**2 + 1e-10)
                    wall_force[close_to_neg] = np.abs(wall_force[close_to_neg])
                
                forces[:, k] += wall_force
            
            # Normalize forces
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes) if np.max(force_magnitudes) > 0 else 1.0
            if max_force > 1.0:
                forces /= max_force
            
            centers += step * forces
            centers = np.clip(centers, -0.4999, 0.4999)
            
            new_r = get_min_radius(centers)
            
            if new_r > current_r:
                step = min(step * 1.05, 0.015)
                no_improve_count = 0
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                step = max(step * 0.95, 0.0005)
                no_improve_count += 1
            
            current_r = new_r
            
            if no_improve_count > max_no_improve:
                n_perturb = max(1, N // 10)
                perturbed = np.random.choice(N, n_perturb, replace=False)
                centers[perturbed] += np.random.randn(n_perturb, 3) * 0.005
                centers = np.clip(centers, -0.4999, 0.4999)
                no_improve_count = 0
                step = 0.003
        
        return best_local_centers, best_local_r

    def optimize_soft(centers, max_steps=10000, beta_start=5.0, beta_end=1000.0, lr_start=0.005, lr_end=0.00005):
        """
        Optimizes using soft-min surrogate with gradient descent.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            progress = step / max_steps
            beta = beta_end - (beta_end - beta_start) * 0.5 * (1 + np.cos(np.pi * progress))
            lr = lr_start * (lr_end / lr_start) ** progress
            
            grad = compute_gradients(centers, beta)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                base_step = lr * max(current_r, 0.01) * 5.0
                best_step = base_step
                best_new_r = -np.inf
                for scale in [1.0, 0.5, 2.0]:
                    trial_centers = centers + base_step * scale * grad / grad_norm
                    trial_centers = np.clip(trial_centers, -0.4999, 0.4999)
                    trial_r = get_min_radius(trial_centers)
                    if trial_r > best_new_r:
                        best_new_r = trial_r
                        best_step = base_step * scale
                
                centers = centers + best_step * grad / grad_norm
                centers = np.clip(centers, -0.4999, 0.4999)
            
            new_r = get_min_radius(centers)
            current_r = new_r
            
            if new_r > best_local_r:
                best_local_r = new_r
                best_local_centers = centers.copy()
            
            if step % 500 == 250 and step > 0:
                noise_scale = lr * 0.1 * max(current_r, 0.01)
                centers += np.random.randn(N, 3) * noise_scale
                centers = np.clip(centers, -0.4999, 0.4999)

        return best_local_centers, best_local_r

    def optimize_scipy(centers, max_time=15.0):
        """
        Use scipy's L-BFGS-B to polish the configuration.
        """
        centers_flat = centers.copy().flatten()
        start_opt = time.time()
        
        beta = 200.0
        
        def objective(x):
            c = x.reshape(N, 3)
            return -soft_min_value(c, beta)
        
        def gradient(x):
            c = x.reshape(N, 3)
            return compute_gradients(c, beta).flatten()
        
        try:
            res = minimize(objective, centers_flat, method='L-BFGS-B', jac=gradient,
                          bounds=[(-0.4999, 0.4999)] * (N*3),
                          options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if time.time() - start_opt > max_time * 0.7:
                new_centers = res.x.reshape(N, 3)
                new_r = get_min_radius(new_centers)
                if new_r > get_min_radius(centers):
                    return new_centers, new_r
                return centers, get_min_radius(centers)
            
            beta = 500.0
            centers_flat = res.x
            
            def objective2(x):
                c = x.reshape(N, 3)
                return -soft_min_value(c, beta)
            
            def gradient2(x):
                c = x.reshape(N, 3)
                return compute_gradients(c, beta).flatten()
            
            res2 = minimize(objective2, centers_flat, method='L-BFGS-B', jac=gradient2,
                          bounds=[(-0.4999, 0.4999)] * (N*3),
                          options={'maxiter': 200, 'maxfun': 1500, 'ftol': 1e-16, 'gtol': 1e-12})
            
            new_centers = res2.x.reshape(N, 3)
            new_r = get_min_radius(new_centers)
            
            if new_r > get_min_radius(centers):
                return new_centers, new_r
            else:
                return centers, get_min_radius(centers)
                
        except Exception:
            return centers, get_min_radius(centers)

    def optimize_hard(centers, max_steps=3000, step_size=0.001):
        """
        Final polish using hard constraint logic.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = step_size
        no_improve_count = 0
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i_min, j_min = min_idx
            min_dist_val = dist[i_min, j_min]
            
            wall_dists = 0.5 - np.abs(centers)
            min_wall_val = np.min(wall_dists)
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i_wall, k_wall = min_wall_idx
            
            r_pair = min_dist_val / 2.0
            r_wall = min_wall_val
            
            forces = np.zeros_like(centers)
            
            if r_pair < r_wall:
                direction = (centers[i_min] - centers[j_min]) / (min_dist_val + 1e-10)
                push = 0.5 * step
                forces[i_min] += direction * push
                forces[j_min] -= direction * push
            else:
                if centers[i_wall, k_wall] > 0:
                    forces[i_wall, k_wall] -= 0.5 * step
                else:
                    forces[i_wall, k_wall] += 0.5 * step
            
            centers += forces
            centers = np.clip(centers, -0.4999, 0.4999)
            
            new_r = get_min_radius(centers)
            
            if new_r > best_local_r + 1e-15:
                best_local_r = new_r
                best_local_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                step *= 0.9
                if step < 1e-10 or no_improve_count > 100:
                    break

        return best_local_centers, best_local_r

    def full_pipeline(init_centers, label=""):
        """Run the full optimization pipeline on an initial configuration."""
        nonlocal best_r, best_centers
        
        c = init_centers.copy()
        r = get_min_radius(c)
        
        # Step 1: Force-directed relaxation (global exploration)
        if time.time() - start_time < time_limit - 30:
            c, r = force_directed_relax(c, max_steps=8000, initial_step=0.003)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        
        # Step 2: Soft-min gradient optimization (refinement)
        if time.time() - start_time < time_limit - 25:
            c, r = optimize_soft(c, max_steps=5000)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        
        # Step 3: Scipy L-BFGS-B polish
        if time.time() - start_time < time_limit - 15:
            c, r = optimize_scipy(c, max_time=12.0)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        
        # Step 4: Hard polish
        if time.time() - start_time < time_limit - 5:
            c, r = optimize_hard(c, max_steps=2000, step_size=0.001)
            if r > best_r:
                best_r = r
                best_centers = c.copy()
        
        return c, r

    # --- Initial Configurations ---
    
    # 1. Cubic Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    init_grid_jittered = init_grid + np.random.randn(N, 3) * 0.001
    
    full_pipeline(init_grid_jittered, "grid")
    
    # 2. FCC-like Initialization
    s = 0.28
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base = np.array([i, j, k]) * s
                pts.append(base)
                pts.append(base + np.array([s/2, s/2, 0]))
                pts.append(base + np.array([s/2, 0, s/2]))
                pts.append(base + np.array([0, s/2, s/2]))
    
    fcc_grid = np.array(pts)
    fcc_grid -= fcc_grid.mean(axis=0)
    extent = np.max(np.abs(fcc_grid))
    if extent > 0.5:
        fcc_grid *= (0.5 / extent)
    
    dists_from_origin = np.linalg.norm(fcc_grid, axis=1)
    order = np.argsort(dists_from_origin)
    init_fcc = fcc_grid[order[:N]]
    
    if time.time() - start_time < time_limit - 40:
        full_pipeline(init_fcc, "fcc")
    
    # 3. Random Initialization
    np.random.seed(123)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))
    
    if time.time() - start_time < time_limit - 40:
        full_pipeline(init_random, "random")
    
    # 4. Second random initialization with different seed
    np.random.seed(777)
    init_random2 = np.random.uniform(-0.4, 0.4, (N, 3))
    
    if time.time() - start_time < time_limit - 40:
        full_pipeline(init_random2, "random2")
    
    # 5. Final polish on the best found
    if best_centers is not None and time.time() - start_time < time_limit - 20:
        c_best = best_centers.copy()
        
        # Another round of force-directed with smaller steps
        c_best, r_best = force_directed_relax(c_best, max_steps=5000, initial_step=0.001)
        if r_best > best_r:
            best_r = r_best
            best_centers = c_best.copy()
        
        # Scipy polish
        if time.time() - start_time < time_limit - 10:
            c_best, r_best = optimize_scipy(c_best, max_time=10.0)
            if r_best > best_r:
                best_r = r_best
                best_centers = c_best.copy()
        
        # Hard polish
        if time.time() - start_time < time_limit - 3:
            c_best, r_best = optimize_hard(c_best, max_steps=3000, step_size=0.0005)
            if r_best > best_r:
                best_r = r_best
                best_centers = c_best.copy()

    if best_centers is None:
        return init_grid

    return best_centers
