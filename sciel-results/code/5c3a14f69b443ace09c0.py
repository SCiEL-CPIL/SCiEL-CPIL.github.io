"""
Vectorized Smooth Surrogate Optimization for Sphere Packing.
Uses soft-min via log-sum-exp with vectorized gradients.
Beta-annealing schedule + epigraph variable + final direct polish.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_radius(positions):
        """Compute the exact minimum radius for given positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (N, 3) or not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0

        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)

        dist_to_walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall_dist = np.min(dist_to_walls)

        return min(min_pair_dist / 2.0, min_wall_dist)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Computes the smooth surrogate objective (-soft_min) and its gradient.
        c_flat: (3N,)
        beta: temperature parameter
        Returns: (value, grad_flat)
        """
        c = c_flat.reshape(N, 3)

        # Pairwise: d_ij / 2
        tri_i, tri_j = np.triu_indices(N, k=1)
        diff = c[tri_i] - c[tri_j]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0

        # Walls: 6 distances per point
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5

        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        g_all = np.concatenate([d_pairs, wall_dists])

        # Soft Min via log-sum-exp
        scaled = -beta * g_all
        max_scaled = np.max(scaled)
        if not np.isfinite(max_scaled):
            return 0.0, np.zeros_like(c_flat)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        val = -log_sum_exp / beta

        # Gradient
        weights = np.exp(scaled - log_sum_exp)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0, np.zeros_like(c_flat)
        weights = weights / w_sum

        grad_c = np.zeros_like(c)
        n_pairs = len(d_pairs)

        safe_d = np.maximum(d_pairs, 1e-12)
        factors = weights[:n_pairs] / (4.0 * safe_d)
        factors = np.where(d_pairs > 1e-12, factors, 0.0)

        weighted_diffs = factors[:, np.newaxis] * diff
        np.add.at(grad_c, tri_i, weighted_diffs)
        np.add.at(grad_c, tri_j, -weighted_diffs)

        w_start = n_pairs
        grad_c[:, 0] -= weights[w_start : w_start + N]
        grad_c[:, 0] += weights[w_start + N : w_start + 2 * N]
        grad_c[:, 1] -= weights[w_start + 2 * N : w_start + 3 * N]
        grad_c[:, 1] += weights[w_start + 3 * N : w_start + 4 * N]
        grad_c[:, 2] -= weights[w_start + 4 * N : w_start + 5 * N]
        grad_c[:, 2] += weights[w_start + 5 * N : w_start + 6 * N]

        return -val, -grad_c.flatten()

    def local_optimize_smooth(c_init, beta, maxiter, time_budget):
        """Run L-BFGS-B with smooth objective."""
        if remaining() < time_budget:
            return c_init

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            val, _ = smooth_objective_and_grad(x, beta)
            return val

        def grad_flat(x):
            _, grad = smooth_objective_and_grad(x, beta)
            return grad

        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), jac=grad_flat, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def local_optimize_direct(c_init, maxiter, time_budget):
        """Run L-BFGS-B on the exact non-smooth objective."""
        if remaining() < time_budget:
            return c_init

        bounds = [(-0.499999, 0.499999)] * (N * 3)

        def obj_flat(x):
            return -compute_radius(x.reshape(N, 3))

        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def make_initial_grid():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        """BCC lattice in a 3x3x3 supercell."""
        positions = []
        a = 1.0 / 3.0
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -0.5 + ix * a
                    y = -0.5 + iy * a
                    z = -0.5 + iz * a
                    positions.append([x, y, z])
                    positions.append([x + a / 2, y + a / 2, z + a / 2])

        positions = np.array(positions, dtype=np.float64)
        mask = np.all(np.abs(positions) <= 0.49, axis=1)
        positions = positions[mask]

        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]

        if len(positions) < N:
            np.random.seed(42)
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return positions[:N]

    def make_fcc_lattice():
        """FCC lattice in a 3x3x3 supercell."""
        positions = []
        a = 1.0 / 3.0
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -0.5 + ix * a
                    y = -0.5 + iy * a
                    z = -0.5 + iz * a
                    positions.append([x, y, z])
                    if ix < 2:
                        positions.append([x + a / 2, y, z])
                    if iy < 2:
                        positions.append([x, y + a / 2, z])
                    if iz < 2:
                        positions.append([x, y, z + a / 2])

        positions = np.array(positions, dtype=np.float64)
        mask = np.all(np.abs(positions) <= 0.49, axis=1)
        positions = positions[mask]

        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]

        if len(positions) < N:
            np.random.seed(42)
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return positions[:N]

    def make_hcp_like():
        """Hexagonal close-packed inspired."""
        positions = []
        a = 0.32
        for z_val in [-0.3, 0.0, 0.3]:
            for row in range(4):
                y = -0.4 + row * a * np.sqrt(3) / 2
                for col in range(4):
                    x = -0.4 + col * a + (row % 2) * a / 2
                    if abs(x) <= 0.48 and abs(y) <= 0.48 and abs(z_val) <= 0.48:
                        positions.append([x, y, z_val])

        positions = np.array(positions, dtype=np.float64)
        mask = np.all(np.abs(positions) <= 0.49, axis=1)
        positions = positions[mask]

        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]

        if len(positions) < N:
            np.random.seed(42)
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])

        return positions[:N]

    def make_random(seed):
        np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    def make_stratified(seed):
        """Stratified sampling for better coverage."""
        np.random.seed(seed)
        m = 4
        grid_pts = []
        for x in range(m):
            for y in range(m):
                for z in range(m):
                    grid_pts.append((x, y, z))
        np.random.shuffle(grid_pts)
        positions = np.zeros((N, 3))
        for i in range(N):
            gx, gy, gz = grid_pts[i]
            positions[i, 0] = -0.5 + (gx + 0.5) / m + np.random.uniform(-0.1, 0.1) / m
            positions[i, 1] = -0.5 + (gy + 0.5) / m + np.random.uniform(-0.1, 0.1) / m
            positions[i, 2] = -0.5 + (gz + 0.5) / m + np.random.uniform(-0.1, 0.1) / m
        return np.clip(positions, -0.49, 0.49)

    best_positions = None
    best_radius = -1.0

    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("fcc", make_fcc_lattice),
        ("hcp", make_hcp_like),
        ("strat0", lambda: make_stratified(0)),
        ("strat1", lambda: make_stratified(7)),
        ("strat2", lambda: make_stratified(99)),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
    ]

    # Phase 1: Global Optimization with Smooth Surrogate
    for name, init_func in initializations:
        if remaining() < 10:
            break

        try:
            c = init_func()
            c = np.clip(c, -0.499999, 0.499999)

            # Beta annealing
            betas = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]

            for beta in betas:
                if remaining() < 4:
                    break
                c = local_optimize_smooth(c, beta, maxiter=200, time_budget=1.5)

            # Direct polish
            if remaining() > 3:
                c = local_optimize_direct(c, maxiter=100, time_budget=3)

            r = compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_positions = c.copy()

        except Exception:
            continue

    # Phase 2: Perturbations of Best Solution
    if best_positions is not None and remaining() > 15:
        for perturb_seed in range(20):
            if remaining() < 5:
                break
            np.random.seed(2000 + perturb_seed)
            scale = 0.005 * (1 + 0.5 * (perturb_seed % 4))
            c_pert = best_positions + np.random.normal(0, scale, size=best_positions.shape)
            c_pert = np.clip(c_pert, -0.499999, 0.499999)

            for beta in [100.0, 300.0, 600.0]:
                if remaining() < 3:
                    break
                c_pert = local_optimize_smooth(c_pert, beta, maxiter=100, time_budget=1.0)

            if remaining() > 2:
                c_pert = local_optimize_direct(c_pert, maxiter=60, time_budget=2)

            r = compute_radius(c_pert)
            if r > best_radius:
                best_radius = r
                best_positions = c_pert.copy()

    # Phase 3: Final high-precision polish
    if best_positions is not None and remaining() > 6:
        for _ in range(5):
            if remaining() < 3:
                break
            if remaining() > 5:
                best_positions = local_optimize_smooth(best_positions, beta=4000.0, maxiter=300, time_budget=1.5)
            best_positions = local_optimize_direct(best_positions, maxiter=300, time_budget=2)
            best_positions = np.clip(best_positions, -0.499999, 0.499999)

    if best_positions is None:
        best_positions = make_initial_grid()

    return best_positions

# EVOLVE-BLOCK-END
