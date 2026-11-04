import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp
import sys

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # Precompute triangle indices once
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def get_radius(c):
        """Compute the exact maximum common radius for centers c (N,3)."""
        c = np.asarray(c, dtype=np.float64)
        if c.shape != (N, 3) or not np.all(np.isfinite(c)):
            return 0.0
        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 0.0
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        return float(min(min_pair_dist / 2.0, min_wall_dist))

    def smooth_objective_and_grad(c_flat, beta):
        """
        Vectorized soft-min surrogate for the packing objective.
        Returns (value, gradient) for minimization of -soft_min(g_k).
        """
        c = c_flat.reshape(N, 3)

        if np.any(np.abs(c) > 0.5 + 1e-9):
            return 1e6, np.zeros_like(c_flat)

        # Pairwise distances
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
        lse = logsumexp(neg_beta_g)
        val = -lse / beta

        # Weights
        weights = np.exp(neg_beta_g - lse)

        # Gradient: vectorized pairwise contribution
        grad_c = np.zeros_like(c)

        diff_ij = c[tri_i] - c[tri_j]
        d_ij = d_full[tri_i, tri_j]
        safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
        unit_vecs = diff_ij / safe_d[:, np.newaxis]
        # Gradient of soft_min(d_pairs/2) w.r.t c_i: sum_j w_j * (c_i - c_j) / (2 * d_ij)
        w_pair = weights[:n_pairs] / (2.0 * safe_d)
        contrib = w_pair[:, np.newaxis] * unit_vecs

        np.add.at(grad_c, tri_i, contrib)
        np.add.at(grad_c, tri_j, -contrib)

        # Wall gradients
        ws = n_pairs
        grad_c[:, 0] -= weights[ws:ws + N]
        grad_c[:, 0] += weights[ws + N:ws + 2 * N]
        grad_c[:, 1] -= weights[ws + 2 * N:ws + 3 * N]
        grad_c[:, 1] += weights[ws + 3 * N:ws + 4 * N]
        grad_c[:, 2] -= weights[ws + 4 * N:ws + 5 * N]
        grad_c[:, 2] += weights[ws + 5 * N:ws + 6 * N]

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
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                         'maxls': 100, 'maxfun': max_iter * 10, 'maxcor': 50}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def run_direct_optimization(c_init, max_iter):
        """Run L-BFGS-B on the exact non-smooth objective."""
        x0 = c_init.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)

        def fun(x):
            return -get_radius(x.reshape(N, 3))

        try:
            result = minimize(
                fun, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def shake_and_bake(c_init, r_current, num_kicks=8, kick_strength=0.03):
        """
        Identify critical pairs and apply repulsive kicks to escape local minima.
        Uses multiple strategies: single-pair kick, multi-pair kick, random perturbation.
        """
        c = c_init.copy()
        best_r = r_current
        best_c = c.copy()
        
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        
        for kick_idx in range(num_kicks):
            if remaining() < 5.0:
                break
            
            # Find the minimum distance pair
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            min_pair_dist = np.min(d)
            min_wall_dist = np.min(0.5 - np.abs(c))
            
            strategy = kick_idx % 3
            
            if strategy == 0:
                # Single closest pair kick
                if min_wall_dist < min_pair_dist / 2.0:
                    # Find point closest to wall
                    wall_dists = np.min(0.5 - np.abs(c), axis=1)
                    idx = np.argmin(wall_dists)
                    c[idx] -= np.sign(c[idx]) * kick_strength
                    c = np.clip(c, -0.499, 0.499)
                else:
                    # Find closest pair
                    idx_flat = np.argmin(d)
                    i, j = np.unravel_index(idx_flat, (N, N))
                    if i == j:
                        continue
                    if i > j:
                        i, j = j, i
                    
                    vec = c[j] - c[i]
                    dist = np.linalg.norm(vec)
                    if dist < 1e-9:
                        vec = np.array([1.0, 0, 0])
                        dist = 1.0
                    else:
                        vec /= dist
                    
                    c[i] -= vec * kick_strength
                    c[j] += vec * kick_strength
                    c = np.clip(c, -0.499, 0.499)
                    
            elif strategy == 1:
                # Multi-pair kick: kick top 3 closest pairs
                d_sorted = d.copy()
                np.fill_diagonal(d_sorted, np.inf)
                # Get top 3 closest pairs
                flat_d = d_sorted.flatten()
                top3_indices = np.argsort(flat_d)[:3]
                for idx_flat in top3_indices:
                    i, j = np.unravel_index(idx_flat, (N, N))
                    if i == j:
                        continue
                    if i > j:
                        i, j = j, i
                    vec = c[j] - c[i]
                    dist = np.linalg.norm(vec)
                    if dist < 1e-9:
                        continue
                    vec /= dist
                    c[i] -= vec * kick_strength * 0.5
                    c[j] += vec * kick_strength * 0.5
                c = np.clip(c, -0.499, 0.499)
                
            else:
                # Random perturbation of a few random points
                n_perturb = rng.randint(2, 5)
                indices = rng.choice(N, size=n_perturb, replace=False)
                for idx in indices:
                    # Perturb in random direction
                    direction = rng.randn(3)
                    direction /= np.linalg.norm(direction)
                    c[idx] += direction * kick_strength * 0.8
                c = np.clip(c, -0.499, 0.499)
            
            # Re-optimize from this kicked position with adaptive beta
            beta_opt = 2000 if strategy == 0 else 1000
            c_new = run_smooth_optimization(c, beta=beta_opt, max_iter=60)
            r_new = get_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                # Reset c to the new best for next kick
                c = best_c.copy()
            
        return best_c, best_r

    def make_fcc_dense_init():
        """
        Dense FCC lattice initialization.
        """
        a = 0.25
        positions = []
        basis = np.array([[0, 0, 0],
                          [0.5, 0.5, 0],
                          [0.5, 0, 0.5],
                          [0, 0.5, 0.5]])

        n_cells = 4
        for iz in range(n_cells):
            for iy in range(n_cells):
                for ix in range(n_cells):
                    origin = np.array([ix, iy, iz]) * a
                    for b in basis:
                        pos = origin + b * a
                        if np.all(np.abs(pos - 0.5) <= 0.48):
                            positions.append(pos)

        positions = np.array(positions, dtype=np.float64)

        if len(positions) >= N:
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(-min_dists)[:N]
            positions = positions[idx]
        else:
            rng = np.random.RandomState(123)
            extra = rng.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return np.clip(positions[:N].copy(), -0.499, 0.499)

    def make_grid_init():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_init():
        """BCC-inspired: body-centered cubic lattice."""
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
                    bcc_points.append([cx + cell_size / 2, cy + cell_size / 2, cz + cell_size / 2])
        bcc_arr = np.array(bcc_points)
        mask = np.all(np.abs(bcc_arr) <= 0.5, axis=1)
        bcc_arr = bcc_arr[mask]
        if len(bcc_arr) >= N:
            return bcc_arr[:N].copy()
        else:
            extra = np.random.RandomState(7).uniform(-0.45, 0.45, size=(N - len(bcc_arr), 3))
            return np.vstack([bcc_arr, extra])

    def make_rotated_random_init(seed):
        """Random points rotated by a random orthogonal matrix."""
        rng = np.random.RandomState(seed)
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                       [0, 1, 0],
                       [-np.sin(phi), 0, np.cos(phi)]])
        R = Ry @ Rx
        base = rng.uniform(-0.35, 0.35, size=(N, 3))
        rotated = base @ R.T
        max_abs = np.max(np.abs(rotated))
        if max_abs > 0:
            rotated = rotated * (0.45 / max_abs)
        return np.clip(rotated, -0.499, 0.499)

    def make_layered_init():
        """Layered initialization: distribute points across 5 layers."""
        n_layers = 5
        layer_height = 0.8 / n_layers
        points_per_layer = N // n_layers
        remaining_pts = N - points_per_layer * n_layers
        c = np.zeros((N, 3))
        idx = 0
        for layer in range(n_layers):
            z = -0.4 + layer_height * (layer + 0.5)
            n_in_layer = points_per_layer + (1 if layer < remaining_pts else 0)
            n_x = int(np.ceil(np.sqrt(n_in_layer)))
            n_y = n_in_layer // n_x + (1 if n_in_layer % n_x else 0)
            x_spacing = 0.8 / max(n_x, 1)
            y_spacing = 0.8 / max(n_y, 1)
            count = 0
            for iy in range(n_y):
                for ix in range(n_x):
                    if count >= n_in_layer:
                        break
                    x = -0.4 + x_spacing * (ix + 0.5)
                    y = -0.4 + y_spacing * (iy + 0.5)
                    c[idx] = [x, y, z]
                    idx += 1
                    count += 1
        return np.clip(c, -0.499, 0.499)

    def make_hcp_init():
        """HCP-inspired initialization."""
        a = 0.3
        c = 0.5
        positions = []
        basis = np.array([[0, 0, 0],
                          [1/3, 2/3, 0.5]])

        n_cells = 4
        for iz in range(n_cells):
            for iy in range(n_cells):
                for ix in range(n_cells):
                    x = a * (ix + iy / 2)
                    y = a * (iy * np.sqrt(3) / 2)
                    z = c * iz
                    for b in basis:
                        pos = np.array([x + a * b[0], y + a * b[1], z + c * b[2]])
                        if np.all(np.abs(pos - 0.5) <= 0.48):
                            positions.append(pos)

        positions = np.array(positions, dtype=np.float64)

        if len(positions) >= N:
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(-min_dists)[:N]
            positions = positions[idx]
        else:
            rng = np.random.RandomState(456)
            extra = rng.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return np.clip(positions[:N].copy(), -0.499, 0.499)

    def feasibility_check(c_init, r_target, max_iter=100):
        """
        Check if radius r_target is feasible by minimizing overlap penalty.
        Returns (is_feasible, best_config, min_penalty)
        """
        x0 = c_init.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)

        def penalty_and_grad(x):
            c = x.reshape(N, 3)
            penalty = 0.0
            grad_c = np.zeros_like(c)
            
            # Pairwise
            diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
            d_pairs = d_full[tri_i, tri_j]
            
            # Violation: 2*r - d > 0
            viol = np.maximum(0, 2 * r_target - d_pairs)
            penalty += np.sum(viol ** 2)
            
            # Gradient of sum(viol^2) w.r.t c
            # d(viol^2)/dc = 2*viol * d(viol)/dc
            # d(2r - d)/dc_i = - (c_i - c_j) / d
            # d(2r - d)/dc_j = + (c_i - c_j) / d
            
            grad_pairs = np.zeros_like(c)
            mask = viol > 0
            if np.any(mask):
                d_safe = np.where(d_pairs[mask] > 1e-12, d_pairs[mask], 1.0)
                coeff = 2 * viol[mask] / d_safe
                vecs = c[tri_i[mask]] - c[tri_j[mask]]
                
                # Contribution to c_i
                np.add.at(grad_pairs, tri_i[mask], -coeff[:, np.newaxis] * vecs)
                # Contribution to c_j
                np.add.at(grad_pairs, tri_j[mask], coeff[:, np.newaxis] * vecs)
            
            # Walls
            # x+ : 0.5 - c_x < r => c_x - 0.5 > -r => viol = max(0, r - (0.5 - c_x)) = max(0, r + c_x - 0.5)
            # Actually, dist_to_wall = 0.5 - |c|. Violation is r - dist > 0.
            # Let's stick to the standard: dist = 0.5 - |c|. viol = max(0, r - dist).
            
            wall_dists = np.min(0.5 - np.abs(c), axis=1)
            wall_viol = np.maximum(0, r_target - wall_dists)
            penalty += np.sum(wall_viol ** 2)
            
            # Gradient for walls
            # d(r - (0.5 - |c_k|))/dc_k = sign(c_k) for the active dimension
            # Only if viol > 0
            mask_w = wall_viol > 0
            if np.any(mask_w):
                # Determine which wall is active for each point
                # This is tricky to vectorize perfectly without knowing which of 6 walls is min.
                # Approximation: use the sign of the coordinate with smallest distance.
                # For simplicity and robustness, we can just push towards center if viol > 0.
                # grad = -2 * viol * d(dist)/dc = -2 * viol * (-sign(c)) = 2 * viol * sign(c)
                # This pushes c towards 0 if c is positive, and away from 0 if c is negative?
                # Wait. dist = 0.5 - |c|. d(dist)/dc = -sign(c).
                # d(viol)/dc = -sign(c) (if viol > 0).
                # d(penalty)/dc = 2 * viol * d(viol)/dc = -2 * viol * sign(c).
                # So if c > 0, grad is negative -> pushes c down (towards center). Correct.
                # If c < 0, grad is positive -> pushes c up (towards center). Correct.
                
                # We need to apply this only to the dimension that is limiting.
                # For now, let's use a simpler heuristic: push towards center proportionally to violation.
                # This is a subgradient.
                for i in np.where(mask_w)[0]:
                    # Find the dimension with min distance
                    dists = 0.5 - np.abs(c[i])
                    dim = np.argmin(dists)
                    # Push towards center
                    grad_c[i, dim] -= 2 * wall_viol[i] * np.sign(c[i, dim])
                    # If c is 0, sign is 0, we need a random kick or just 0.
                    if c[i, dim] == 0:
                        grad_c[i, dim] -= 2 * wall_viol[i] * 0.1 # small push
            
            grad_c += grad_pairs
            return penalty, grad_c.flatten()

        def fun(x):
            p, _ = penalty_and_grad(x)
            return p

        def jac(x):
            _, g = penalty_and_grad(x)
            return g

        try:
            result = minimize(
                fun, x0, jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            min_penalty = result.fun
            is_feasible = min_penalty < 1e-6
            return is_feasible, np.clip(result.x.reshape(N, 3), -0.499, 0.499), min_penalty
        except Exception:
            return False, c_init.copy(), 1e6

    def bisection_refinement(c_init, r_low, r_high, max_iter_per_check=50):
        """
        Use bisection on radius to find the maximum feasible radius.
        """
        best_c = c_init.copy()
        best_r = r_low
        min_penalty = 1e6
        
        # First, check if r_high is feasible to set a valid upper bound
        # If not, we might need to lower it, but we assume r_low is feasible.
        
        for _ in range(20): # ~20 iterations for bisection
            if remaining() < 5.0:
                break
            
            r_mid = (r_low + r_high) / 2.0
            is_feasible, c_mid, p_mid = feasibility_check(best_c, r_mid, max_iter=max_iter_per_check)
            
            if is_feasible:
                # r_mid is feasible, try higher
                r_low = r_mid
                best_c = c_mid.copy()
                best_r = r_mid
                min_penalty = p_mid
            else:
                # r_mid is not feasible, try lower
                r_high = r_mid
        
        return best_c, best_r

    best_positions = None
    best_radius = -1.0

    def update_best(pos):
        nonlocal best_positions, best_radius
        r = get_radius(pos)
        if r > best_radius:
            best_radius = r
            best_positions = pos.copy()
        return r

    # Define initializations
    initializations = [
        ("fcc_dense", make_fcc_dense_init),
        ("grid", make_grid_init),
        ("bcc", make_bcc_init),
        ("hcp", make_hcp_init),
        ("layered", make_layered_init),
        ("rot0", lambda: make_rotated_random_init(0)),
        ("rot1", lambda: make_rotated_random_init(42)),
        ("rot2", lambda: make_rotated_random_init(123)),
        ("rot3", lambda: make_rotated_random_init(777)),
    ]

    # Stage 1: Global Search with Soft-Min Annealing
    for name, init_func in initializations:
        if remaining() < 10:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499, 0.499)

            # Simple annealing schedule
            betas = [100, 500, 2000, 10000, 50000]
            c_best = positions
            r_best = get_radius(c_best)
            
            for beta in betas:
                if remaining() < 3.0:
                    break
                c_new = run_smooth_optimization(c_best, beta, max_iter=100)
                r_new = get_radius(c_new)
                if r_new > r_best:
                    r_best = r_new
                    c_best = c_new.copy()
            
            # Shake and bake to escape local minima
            c_shaken, r_shaken = shake_and_bake(c_best, r_best)
            
            # Update global best
            r_updated = update_best(c_shaken)
            
            # Stage 2: Bisection Refinement on the best candidate found so far
            if best_positions is not None and remaining() > 15:
                # Estimate upper bound: current best + 5%
                r_upper = best_radius * 1.05
                r_lower = best_radius * 0.95
                c_refined, r_refined = bisection_refinement(best_positions, r_lower, r_upper)
                if r_refined > best_radius:
                    update_best(c_refined)

        except Exception:
            continue

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499, 0.499)
    else:
        best_positions = make_grid_init()
        best_positions = np.clip(best_positions, -0.499, 0.499)

    # Final high-precision polish if time remains
    if remaining() > 6:
        try:
            # Use a very high beta for final polish
            polished = run_smooth_optimization(best_positions, beta=500000, max_iter=200)
            polished = run_direct_optimization(polished, max_iter=150)
            r_polished = get_radius(polished)
            if r_polished > best_radius:
                best_positions = np.clip(polished, -0.499, 0.499)
                best_radius = r_polished
        except Exception:
            pass
    
    # Additional polish: try a few more shake-and-bake cycles if time allows
    if remaining() > 10 and best_positions is not None:
        try:
            c_final, r_final = shake_and_bake(best_positions, best_radius, num_kicks=4, kick_strength=0.02)
            r_final = get_radius(c_final)
            if r_final > best_radius:
                best_positions = np.clip(c_final, -0.499, 0.499)
                best_radius = r_final
        except Exception:
            pass

    return best_positions
