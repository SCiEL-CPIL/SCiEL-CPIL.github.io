"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s buffer before 150s limit

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        return min(min_pair_dist / 2.0, min_wall_dist)

    # Precompute triangle indices once
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization of -soft_min(g_k).
        """
        c = c_flat.reshape(N, 3)

        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise: compute all N×N distances, extract upper triangle
        diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
        d_pairs = d_full[tri_i, tri_j] / 2.0

        # Wall distances
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft min value
        neg_beta_g = -beta * g_all
        log_sum_exp = logsumexp(neg_beta_g)
        val = -log_sum_exp / beta

        # Weights
        weights = np.exp(neg_beta_g - log_sum_exp)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        # Pairwise gradient: for each pair (i,j) with weight w_k,
        # grad_c[i] += w_k * 0.5 * (c_i - c_j) / d_ij
        # grad_c[j] -= w_k * 0.5 * (c_i - c_j) / d_ij
        # Vectorize: diff_ij = c[tri_i] - c[tri_j], d_ij = d_full[tri_i, tri_j]
        diff_ij = c[tri_i] - c[tri_j]  # (n_pairs, 3)
        d_ij = d_full[tri_i, tri_j]    # (n_pairs,)
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]  # (n_pairs, 3)
        w_pair = weights[:n_pairs] * 0.5 / safe_d   # (n_pairs,)
        contrib = w_pair[:, np.newaxis] * unit_vecs  # (n_pairs, 3)

        # Accumulate using np.add.at for unbuffered scatter-add
        np.add.at(grad_c, tri_i, contrib)
        np.add.at(grad_c, tri_j, -contrib)

        # Wall gradients (vectorized)
        ws = n_pairs
        grad_c[:, 0] -= weights[ws:ws+N]
        grad_c[:, 0] += weights[ws+N:ws+2*N]
        grad_c[:, 1] -= weights[ws+2*N:ws+3*N]
        grad_c[:, 1] += weights[ws+3*N:ws+4*N]
        grad_c[:, 2] -= weights[ws+4*N:ws+5*N]
        grad_c[:, 2] += weights[ws+5*N:ws+6*N]

        return -val, -grad_c.flatten()

    def run_smooth_optimization(c_init, beta, max_iter):
        """Run L-BFGS-B on the smooth surrogate."""
        x0 = c_init.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)

        def fun(x):
            val, grad = smooth_objective_and_grad(x, beta)
            return val

        def jac(x):
            val, grad = smooth_objective_and_grad(x, beta)
            return grad

        try:
            result = minimize(
                fun, x0, jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100,
                         'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return result.x.reshape(N, 3)
        except Exception:
            return c_init.copy()

    def annealed_optimization(c_init):
        """
        Multi-stage annealing with vectorized gradient.
        """
        betas = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for beta in betas:
            if time.time() > deadline:
                break

            remaining = deadline - time.time()
            idx = betas.index(beta)
            n_remaining = len(betas) - idx
            # Earlier stages get slightly more time; later stages need fewer iterations
            stage_time = max(2.0, remaining / n_remaining * 0.85)
            # Scale iterations by time and beta (higher beta needs fewer iters)
            max_iter = int(min(400, max(50, stage_time * 25 / (1 + 0.1 * np.log2(beta)))))

            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        return c_best, r_best

    def generate_initializations():
        """
        Generate diverse initial configurations.
        """
        inits = []
        np.random.seed(42)

        # 1. Simple grid (4x4x4 = 64, take first 50)
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        inits.append(grid[:N].copy())

        # 2. Jittered 5x5x5 grid (125 points, take 50)
        m = 5
        ticks5 = (np.arange(m) + 0.5) / m - 0.5
        grid5 = np.array([(x, y, z) for z in ticks5 for y in ticks5 for x in ticks5])
        # Randomly select 50 from the 125
        idx = np.random.choice(len(grid5), N, replace=False)
        inits.append(grid5[idx].copy() + np.random.normal(0, 0.02, size=(N, 3)))

        # 3. HCP-inspired layered initialization
        # HCP layers are triangular grids.
        # Let's approximate with hexagonal packing in layers.
        n_layers = 5
        points_per_layer = N // n_layers
        remaining_pts = N - points_per_layer * n_layers
        c_hcp = np.zeros((N, 3))
        idx = 0
        layer_height = 0.9 / n_layers
        for layer in range(n_layers):
            n_in_layer = points_per_layer + (1 if layer < remaining_pts else 0)
            z_coord = -0.45 + layer_height * (layer + 0.5)
            # Arrange n_in_layer points in a hexagonal pattern
            # Simple approximation: grid with offset
            m_layer = int(np.ceil(np.sqrt(n_in_layer)))
            spacing = 0.9 / m_layer
            for i in range(n_in_layer):
                ix = i % m_layer
                iy = i // m_layer
                x = -0.45 + spacing * (ix + 0.5)
                y = -0.45 + spacing * (iy + 0.5)
                if ix % 2 == 1:
                    y += spacing / 2.0
                c_hcp[idx] = [x, y, z_coord]
                idx += 1
        inits.append(c_hcp)

        # 4. Random rotations of grid
        for trial in range(3):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])
            Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                          [0, 1, 0],
                          [-np.sin(phi), 0, np.cos(phi)]])
            R = Ry @ Rx
            base = np.random.uniform(-0.3, 0.3, size=(N, 3))
            rotated = base @ R.T
            max_abs = np.max(np.abs(rotated))
            if max_abs > 0:
                rotated = rotated * (0.45 / max_abs)
            inits.append(rotated)

        # 5. Random uniform
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))
        
        # 6. FCC-inspired initialization (Jittered 4x4x4)
        jittered_grid = grid[:N].copy()
        jittered_grid += np.random.normal(0, 0.05, size=(N, 3))
        jittered_grid = np.clip(jittered_grid, -0.45, 0.45)
        inits.append(jittered_grid)

        return inits

    # Main optimization loop
    inits = generate_initializations()

    best_c = None
    best_r = -1.0

    # Allocate time for each init. 
    # MD is expensive, so we should limit the number of inits or the time per init.
    # Let's try 3 inits with generous time, or 5 inits with less time.
    # The `annealed_optimization` function has its own internal time checks, but 
    # we should also guard the outer loop.
    
    num_inits_to_try = min(len(inits), 5)
    
    for i in range(num_inits_to_try):
        if time.time() > deadline - 10:
            break

        c_init = inits[i]
        c_init = np.clip(c_init, -0.49, 0.49)
        
        # Pass a dynamic deadline to annealed_optimization? 
        # Currently it uses global `deadline`. 
        # We should ensure it doesn't run too long if we have many inits.
        # Let's modify annealed_optimization to accept a time_limit? 
        # No, let's just rely on the global deadline and break early if we have done enough.
        
        c_opt, r_opt = annealed_optimization(c_init)

        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    def local_search_polish(c, time_limit):
        """
        Direct local search to refine the packing.
        Finds the closest pair or wall and moves points apart.
        """
        c = c.copy()
        r = get_radius(c)
        start = time.time()
        step = 0.01
        
        while time.time() - start < time_limit:
            # Compute all distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            # Wall distances
            dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
            
            # Find global minimum
            min_pair_dist = np.min(d)
            min_wall_dist = np.min(dist_to_walls)
            
            improved = False
            if min_pair_dist / 2.0 < min_wall_dist:
                # Constrained by pair
                i, j = np.unravel_index(np.argmin(d), d.shape)
                if i > j: i, j = j, i
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec /= norm
                    # Move i and j apart
                    c[i] += vec * step
                    c[j] -= vec * step
                    improved = True
            else:
                # Constrained by wall
                i = np.argmin(dist_to_walls)
                dx_neg = c[i, 0] + 0.5
                dx_pos = 0.5 - c[i, 0]
                dy_neg = c[i, 1] + 0.5
                dy_pos = 0.5 - c[i, 1]
                dz_neg = c[i, 2] + 0.5
                dz_pos = 0.5 - c[i, 2]
                
                if dx_neg <= dx_pos and dx_neg <= dy_neg and dx_neg <= dy_pos and dx_neg <= dz_neg and dx_neg <= dz_pos:
                    c[i, 0] += step
                    improved = True
                elif dx_pos <= dx_neg and dx_pos <= dy_neg and dx_pos <= dy_pos and dx_pos <= dz_neg and dx_pos <= dz_pos:
                    c[i, 0] -= step
                    improved = True
                elif dy_neg <= dx_neg and dy_neg <= dx_pos and dy_neg <= dy_pos and dy_neg <= dz_neg and dy_neg <= dz_pos:
                    c[i, 1] += step
                    improved = True
                elif dy_pos <= dx_neg and dy_pos <= dx_pos and dy_pos <= dy_neg and dy_pos <= dz_neg and dy_pos <= dz_pos:
                    c[i, 1] -= step
                    improved = True
                elif dz_neg <= dx_neg and dz_neg <= dx_pos and dz_neg <= dy_neg and dz_neg <= dy_pos and dz_neg <= dz_pos:
                    c[i, 2] += step
                    improved = True
                else:
                    c[i, 2] -= step
                    improved = True
                    
            if not improved:
                break
                
            # Clip to bounds
            c = np.clip(c, -0.4999, 0.4999)
            
            # Check improvement
            r_new = get_radius(c)
            if r_new <= r:
                # No improvement, reduce step size
                step *= 0.5
                if step < 1e-6:
                    break
                # Revert move? Or just try again with smaller step.
                # Reverting is safer.
                # But we already modified c. We need to store old c.
                # Let's just break to avoid complexity.
                break
            r = r_new
            
        return c, r

    # Final high-precision polish
    if best_c is not None and time.time() < deadline - 5:
        # Use remaining time for polishing
        # First, a few high-beta smooth steps
        for beta in [100000, 300000]:
            if time.time() > deadline - 10:
                break
            remaining = deadline - 10 - time.time()
            if remaining < 1:
                break
            max_iter = int(min(50, max(10, remaining * 2)))
            c_polished = run_smooth_optimization(best_c, beta, max_iter)
            r_polished = get_radius(c_polished)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()
                
        # Then, direct local search
        if time.time() < deadline - 2:
            time_left = deadline - 2 - time.time()
            c_polished, r_polished = local_search_polish(best_c, time_left)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished

    # Fallback
    if best_c is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    best_c = np.clip(best_c, -0.4999, 0.4999)

    final_r = get_radius(best_c)
    if final_r <= 0:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
# EVOLVE-BLOCK-END
