"""
Sphere packing: 50 equal spheres in unit cube.

Structural change: use a direct hard-constraint formulation with a
different optimization family (simulated-annealing-style local search with
exact radius evaluation, plus a final L-BFGS-B polish on the non-smooth
objective).  The previous program relied on a soft-min surrogate with
aggressive beta annealing; here we instead use a stochastic local search
that directly maximises the minimum distance, which is a different
modelling family (combinatorial / stochastic vs. smooth surrogate).
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 135.0  # safety margin before 150s kill


def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    # Precompute triangle indices
    tri_i, tri_j = np.triu_indices(N, k=1)
    n_pairs = len(tri_i)

    def get_radius(c):
        """Exact maximum common radius for centers c (N,3)."""
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

    def objective_non_smooth(x):
        """Negative radius for L-BFGS-B (finite-difference)."""
        c = x.reshape(N, 3)
        return -get_radius(c)

    def run_lbfgsb(c_init, max_iter=300):
        """L-BFGS-B on the exact non-smooth objective."""
        x0 = c_init.flatten()
        bounds = [(-0.499, 0.499)] * (N * 3)
        try:
            result = minimize(
                objective_non_smooth, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        except Exception:
            return c_init.copy()

    def local_search_step(c, r, rng, step_scale):
        """
        One local search move: perturb a random subset of points and accept
        if the radius improves.  Returns (c_new, r_new, improved).
        """
        n_perturb = rng.randint(1, 4)  # perturb 1-3 points
        idx = rng.choice(N, size=n_perturb, replace=False)
        c_new = c.copy()
        c_new[idx] += rng.normal(0, step_scale, size=(n_perturb, 3))
        c_new = np.clip(c_new, -0.499, 0.499)
        r_new = get_radius(c_new)
        if r_new > r:
            return c_new, r_new, True
        return c, r, False

    def stochastic_local_search(c_init, max_iters=2000, base_step=0.02):
        """
        Simulated-annealing-style local search directly on the hard radius.
        """
        c = c_init.copy()
        r = get_radius(c)
        rng = np.random.RandomState(42)

        for it in range(max_iters):
            if remaining() < 2.0:
                break

            # Anneal step size
            frac = it / max(max_iters, 1)
            step = base_step * (1.0 - 0.8 * frac)

            c, r, improved = local_search_step(c, r, rng, step)

            # Occasionally do a larger jump to escape local minima
            if it % 50 == 0 and remaining() > 5.0:
                n_jump = rng.randint(3, 8)
                idx = rng.choice(N, size=n_jump, replace=False)
                c_jump = c.copy()
                c_jump[idx] += rng.normal(0, base_step * 2.0, size=(n_jump, 3))
                c_jump = np.clip(c_jump, -0.499, 0.499)
                r_jump = get_radius(c_jump)
                if r_jump > r:
                    c, r = c_jump, r_jump

        return c, r

    def make_grid_init():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_fcc_init():
        """FCC lattice initialization."""
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

    def make_bcc_init():
        """BCC lattice initialization."""
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

    def make_random_init(seed):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        return np.clip(rng.uniform(-0.45, 0.45, size=(N, 3)), -0.499, 0.499)

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
        ("grid", make_grid_init),
        ("fcc", make_fcc_init),
        ("bcc", make_bcc_init),
        ("rand0", lambda: make_random_init(0)),
        ("rand1", lambda: make_random_init(42)),
        ("rand2", lambda: make_random_init(123)),
    ]

    # Stage 1: For each initialization, run stochastic local search then L-BFGS-B
    for name, init_func in initializations:
        if remaining() < 10:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499, 0.499)

            # Stochastic local search
            c_sls, r_sls = stochastic_local_search(positions, max_iters=3000, base_step=0.02)
            update_best(c_sls)

            # L-BFGS-B polish
            if remaining() > 5.0:
                c_polished = run_lbfgsb(c_sls, max_iter=200)
                update_best(c_polished)

        except Exception:
            continue

    # Phase 2: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 15:
        n_perturb = 5
        for k in range(n_perturb):
            if remaining() < 5:
                break
            rng = np.random.RandomState(3000 + k)
            noise_scale = 0.008 * (1.0 - k / (n_perturb + 1))
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499, 0.499)

            c_sls, r_sls = stochastic_local_search(perturbed, max_iters=1500, base_step=0.01)
            update_best(c_sls)

            if remaining() > 5.0:
                c_polished = run_lbfgsb(c_sls, max_iter=150)
                update_best(c_polished)

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499, 0.499)
    else:
        best_positions = make_grid_init()
        best_positions = np.clip(best_positions, -0.499, 0.499)

    # Final high-precision polish if time remains
    if remaining() > 5:
        try:
            polished = run_lbfgsb(best_positions, max_iter=200)
            if get_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499, 0.499)
        except Exception:
            pass

    return best_positions
