"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer before 150s limit

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        # Check validity
        if not np.all(np.isfinite(c)):
            return -np.inf
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return -np.inf
        
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
        # Clip to avoid overflow in exp
        neg_beta_g = np.clip(neg_beta_g, -700, 700)
        log_sum_exp = logsumexp(neg_beta_g)
        val = -log_sum_exp / beta

        # Weights
        weights = np.exp(neg_beta_g - log_sum_exp)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        # Pairwise gradient
        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        w_pair = weights[:n_pairs] * 0.5 / safe_d
        contrib = w_pair[:, np.newaxis] * unit_vecs

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

    def local_refinement(c, iterations=50):
        """
        Simple local search: repeatedly find the closest pair or wall violation
        and push points apart or away from walls.
        """
        c = c.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            
            # Compute distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            # Find min pair
            min_idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = min_idx
            min_d = d[i, j]
            
            # Check walls
            wall_dists = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
            min_wall_idx = np.argmin(wall_dists)
            min_wall_d = wall_dists[min_wall_idx]
            
            # Determine bottleneck
            # If wall is tighter, push away from wall
            if min_wall_d < min_d / 2.0:
                # Find which wall is closest
                c_pt = c[min_wall_idx]
                walls = [
                    (0.5 - c_pt[0], 0, 1),   # +x
                    (c_pt[0] + 0.5, 0, -1),  # -x
                    (0.5 - c_pt[1], 1, 1),   # +y
                    (c_pt[1] + 0.5, 1, -1),  # -y
                    (0.5 - c_pt[2], 2, 1),   # +z
                    (c_pt[2] + 0.5, 2, -1)   # -z
                ]
                wall_d, axis, sign = min(walls)
                # Move slightly away
                step = 0.001
                c[min_wall_idx, axis] += step * sign
            else:
                # Push i and j apart
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec = vec / norm
                    step = 0.001
                    c[i] += step * vec
                    c[j] -= step * vec
            
            # Ensure inside bounds
            c = np.clip(c, -0.499, 0.499)
            
        return c

    def annealed_optimization(c_init):
        """
        Multi-stage annealing with vectorized gradient.
        """
        betas = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000]

        c_best = c_init.copy()
        r_best = get_radius(c_init)

        for beta in betas:
            if time.time() > deadline - 10:
                break

            remaining = deadline - time.time()
            idx = betas.index(beta)
            n_remaining = len(betas) - idx
            stage_time = max(2.0, remaining / n_remaining * 0.85)
            max_iter = int(min(400, max(50, stage_time * 25 / (1 + 0.1 * np.log2(beta)))))

            c_new = run_smooth_optimization(c_best, beta, max_iter)
            r_new = get_radius(c_new)
            if r_new > r_best:
                r_best = r_new
                c_best = c_new.copy()

        # Local refinement
        c_refined = local_refinement(c_best, iterations=100)
        r_refined = get_radius(c_refined)
        if r_refined > r_best:
            r_best = r_refined
            c_best = c_refined.copy()

        return c_best, r_best

    def generate_initializations():
        """
        Generate diverse initial configurations.
        """
        inits = []

        # 1. Simple grid (4x4x4 = 64, take first 50)
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        inits.append(grid[:N].copy())

        # 2. BCC-inspired: body-centered cubic lattice
        n_cell = 4
        cell_size = 1.0 / n_cell
        bcc_points = []
        for iz in range(n_cell):
            for iy in range(n_cell):
                for ix in range(n_cell):
                    cx = -0.5 + cell_size * (ix + 0.5)
                    cy = -0.5 + cell_size * (iy + 0.5)
                    cz = -0.5 + cell_size * (iz + 0.5)
                    bcc_points.append([cx, cy, cz])
                    bcc_points.append([cx + cell_size/2, cy + cell_size/2, cz + cell_size/2])
        bcc_arr = np.array(bcc_points)
        mask = np.all(np.abs(bcc_arr) <= 0.5, axis=1)
        bcc_arr = bcc_arr[mask]
        if len(bcc_arr) >= N:
            inits.append(bcc_arr[:N].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(bcc_arr), 3))
            inits.append(np.vstack([bcc_arr, extra]))

        # 3. FCC-inspired: face-centered cubic lattice
        n_cell = 4
        cell_size = 1.0 / n_cell
        fcc_points = []
        for iz in range(n_cell):
            for iy in range(n_cell):
                for ix in range(n_cell):
                    cx = -0.5 + cell_size * (ix + 0.5)
                    cy = -0.5 + cell_size * (iy + 0.5)
                    cz = -0.5 + cell_size * (iz + 0.5)
                    fcc_points.append([cx, cy, cz])
                    # Face centers
                    if ix < n_cell - 1: fcc_points.append([cx + cell_size/2, cy, cz])
                    if iy < n_cell - 1: fcc_points.append([cx, cy + cell_size/2, cz])
                    if iz < n_cell - 1: fcc_points.append([cx, cy, cz + cell_size/2])
        fcc_arr = np.array(fcc_points)
        mask = np.all(np.abs(fcc_arr) <= 0.5, axis=1)
        fcc_arr = fcc_arr[mask]
        if len(fcc_arr) >= N:
            inits.append(fcc_arr[:N].copy())
        else:
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(fcc_arr), 3))
            inits.append(np.vstack([fcc_arr, extra]))

        # 4. Random rotations of grid
        np.random.seed(42)
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

        # 5. Layered initialization
        n_layers = 5
        layer_height = 0.8 / n_layers
        points_per_layer = N // n_layers
        remaining_pts = N - points_per_layer * n_layers
        c_layered = np.zeros((N, 3))
        pt_idx = 0
        for layer in range(n_layers):
            z = -0.4 + layer * layer_height
            n_pts = points_per_layer + (1 if layer < remaining_pts else 0)
            n_x = int(np.ceil(np.sqrt(n_pts)))
            n_y = int(np.ceil(n_pts / n_x))
            spacing_x = 0.7 / (n_x + 0.5)
            spacing_y = 0.7 / (n_y + 0.5)
            for ix in range(n_x):
                for iy in range(n_y):
                    if pt_idx >= N:
                        break
                    x = -0.35 + ix * spacing_x
                    y = -0.35 + iy * spacing_y
                    if (ix % 2 == 1):
                        y += spacing_y / 2
                    c_layered[pt_idx] = [x, y, z]
                    pt_idx += 1
            if pt_idx >= N:
                break
        inits.append(c_layered)

        # 6. Random uniform initializations
        for seed in [1, 2, 3]:
            np.random.seed(seed)
            inits.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))

        return inits

    # Generate multiple initial configurations
    initial_configs = generate_initializations()

    best_centers = None
    best_r = -np.inf

    # Try each initial configuration
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break

        # First, run annealed smooth optimization
        c_opt, r_opt = annealed_optimization(init)

        if r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt.copy()

    # Final high-precision polish with very high beta
    if best_centers is not None and time.time() < deadline - 5:
        beta_final = 500000
        c_polished = run_smooth_optimization(best_centers, beta_final, 200)
        r_polished = get_radius(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_centers = c_polished.copy()
        
        # One more local refinement
        c_refined = local_refinement(best_centers, iterations=50)
        r_refined = get_radius(c_refined)
        if r_refined > best_r:
            best_r = r_refined
            best_centers = c_refined.copy()

    # If no valid result, fall back to simple grid
    if best_centers is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
