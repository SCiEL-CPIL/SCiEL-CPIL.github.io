"""Circle packing: hard-disk expansion + SLSQP epigraph polish."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    return min(r_wall, r_pair)


def _init_hexagonal():
    """Proper hexagonal lattice: 8 rows x 7 cols, pick 50."""
    positions = []
    # Start with a dense hex grid
    spacing = 0.13
    row_spacing = spacing * math.sqrt(3) / 2
    for row in range(10):
        for col in range(10):
            x = (col - 4.5) * spacing + (0.5 * spacing if row % 2 else 0.0)
            y = (row - 4.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions)
    # Scale to fit in [-0.5, 0.5]
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    # Select 50 closest to center
    dists_to_center = np.sqrt((positions ** 2).sum(axis=1))
    idx = np.argsort(dists_to_center)[:N]
    return positions[idx]


def _init_grid():
    """Grid initialization: 8x8, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()


def _init_jittered_hex(rng):
    """Hexagonal with small random jitter."""
    c = _init_hexagonal().copy()
    c += rng.uniform(-0.005, 0.005, c.shape)
    c = np.clip(c, -0.45, 0.45)
    return c


def _expand_radius(c, r_start, r_end, step, max_iters_per_step=200, t0=None, deadline=None):
    """
    Hard-disk radius expansion: incrementally grow R, resolving overlaps
    with iterative pairwise separation at each step.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _score(c)
    R = r_start

    while R <= r_end:
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break

        # Resolve overlaps at current R
        for _ in range(max_iters_per_step):
            if t0 is not None and deadline is not None:
                if time.time() - t0 > deadline - 3:
                    break

            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.sqrt((diff ** 2).sum(axis=-1))
            np.fill_diagonal(d, 1e-12)

            # Overlap correction
            overlap = np.maximum(2 * R - d, 0.0)
            np.fill_diagonal(overlap, 0.0)

            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    if overlap[i, j] > 1e-10:
                        direction = diff[i, j] / d[i, j]
                        correction = 0.5 * overlap[i, j] * direction
                        c[i] += correction
                        c[j] -= correction
                        moved = True

            # Wall constraints
            for i in range(N):
                for axis in range(2):
                    if c[i, axis] > 0.5 - R:
                        c[i, axis] = 0.5 - R
                        moved = True
                    elif c[i, axis] < -0.5 + R:
                        c[i, axis] = -0.5 + R
                        moved = True

            if not moved:
                break

        c = np.clip(c, -0.5 + R, 0.5 - R)

        r = _score(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

        R += step

    return best_c, best_r


def _slsqp_polish(c, r_init, t0, deadline, max_iter=500):
    """
    SLSQP epigraph: maximize r subject to pairwise distance >= 2r and wall constraints.
    Variables: x = [c_1x, c_1y, ..., c_Nx, c_Ny, r]
    """
    if t0 is not None and time.time() > deadline - 10:
        return c, r_init

    n = N
    x0 = np.concatenate([c.flatten(), [r_init]])

    def objective(x):
        # Minimize -r (last variable)
        return -x[-1]

    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    # Inequality constraints: for each pair, dist >= 2r
    # scipy SLSQP expects constraint >= 0
    constraints = []
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))

    def make_pair_constraint(i, j):
        def fun(x):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            return np.linalg.norm(ci - cj) - 2 * r
        def fun_grad(x):
            g = np.zeros_like(x)
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            d = ci - cj
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                g[2*i:2*i+2] = d / dist
                g[2*j:2*j+2] = -d / dist
            g[-1] = -2.0
            return g
        return {'type': 'ineq', 'fun': fun, 'jac': fun_grad}

    for (i, j) in pair_indices:
        constraints.append(make_pair_constraint(i, j))

    # Wall constraints: 0.5 - |c_ix| >= r, etc.
    for i in range(n):
        for axis in range(2):
            # 0.5 - c_i_axis >= r  =>  0.5 - c_i_axis - r >= 0
            idx = 2 * i + axis
            def wall_con(x, idx=idx):
                return 0.5 - x[idx] - x[-1]
            def wall_con_grad(x, idx=idx):
                g = np.zeros_like(x)
                g[idx] = -1.0
                g[-1] = -1.0
                return g
            constraints.append({'type': 'ineq', 'fun': wall_con, 'jac': wall_con_grad})

            # -0.5 + c_i_axis >= -r  =>  c_i_axis + r - 0.5 >= -1  ... 
            # Actually: c_i_axis >= -0.5 + r  =>  c_i_axis - r + 0.5 >= 0
            def wall_con2(x, idx=idx):
                return x[idx] - x[-1] + 0.5
            def wall_con2_grad(x, idx=idx):
                g = np.zeros_like(x)
                g[idx] = 1.0
                g[-1] = -1.0
                return g
            constraints.append({'type': 'ineq', 'fun': wall_con2, 'jac': wall_con2_grad})

    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.01, 0.1)]

    try:
        result = minimize(
            objective, x0, method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-14}
        )
        polished = result.x[:2*n].reshape(n, 2)
        r_val = result.x[-1]
        # Verify
        polished = np.clip(polished, -0.5, 0.5)
        actual_r = _score(polished)
        if actual_r > r_init:
            return polished, actual_r
    except Exception:
        pass

    return c, r_init


def _local_search(c, r, t0, deadline, n_iterations=500):
    """
    Local search: try small random perturbations and keep improvements.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        if t0 is not None and time.time() - t0 > deadline - 3:
            break

        # Random perturbation
        scale = 0.002 * (1.0 + rng.random())
        perturbation = rng.normal(0, scale, c.shape)
        candidate = c + perturbation
        candidate = np.clip(candidate, -0.45, 0.45)

        r_new = _score(candidate)
        if r_new > best_r:
            best_r = r_new
            best_c = candidate.copy()
            c = candidate.copy()
        # Even if not better, sometimes accept (simulated annealing-like)
        elif rng.random() < 0.1:
            c = candidate.copy()

    return best_c, best_r


def _newton_refine(c, r, t0, deadline, max_iter=100):
    """
    Newton-style refinement: for circles in contact, solve for exact positions
    that maximize the minimum distance.
    """
    if t0 is not None and time.time() - t0 > deadline - 3:
        return c

    c = c.copy()
    tol = 1e-9

    for _ in range(max_iter):
        if t0 is not None and time.time() - t0 > deadline - 3:
            break

        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)

        # Wall distances
        wall_dist = np.minimum(
            0.5 - np.abs(c[:, 0]),
            0.5 - np.abs(c[:, 1])
        )

        # Find minimum
        min_pair = dist.min()
        min_wall = wall_dist.min()
        min_val = min(min_pair, min_wall)

        if min_val <= 2 * r + tol:
            # Try to push the minimum apart
            if min_pair <= min_wall:
                # Find the pair
                idx = np.unravel_index(dist.argmin(), dist.shape)
                i, j = idx[0], idx[1]
                d = c[i] - c[j]
                dist_ij = np.linalg.norm(d)
                if dist_ij > 1e-12:
                    direction = d / dist_ij
                    # Push apart
                    correction = 0.001 * direction
                    c[i] += correction
                    c[j] -= correction
            else:
                # Find the wall-constrained circle
                idx = np.argmin(wall_dist)
                # Push away from wall
                for axis in range(2):
                    if c[idx, axis] > 0:
                        c[idx, axis] -= 0.001
                    else:
                        c[idx, axis] += 0.001

        c = np.clip(c, -0.5, 0.5)

    return c


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # Multiple diverse initializations
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_jittered_hex(rng),
    ]

    best_c = None
    best_r = -1.0

    # Phase 1: Hard-disk radius expansion from each init
    n_inits = len(inits)
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break

        # Expand from 0.06 to 0.0715
        c, r = _expand_radius(start, 0.06, 0.0715, 0.0003, 150, t0, DEADLINE)

        if r > best_r:
            best_r = r
            best_c = c.copy()

    if best_c is None:
        best_c = _init_hexagonal()
        best_r = _score(best_c)

    # Phase 2: SLSQP epigraph polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        c_polished, r_polished = _slsqp_polish(best_c, best_r, t0, DEADLINE, max_iter=300)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished

    # Phase 3: Local search
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c_ls, r_ls = _local_search(best_c, best_r, t0, DEADLINE, n_iterations=300)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls

    # Phase 4: Newton refinement
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        c_newton = _newton_refine(best_c, best_r, t0, DEADLINE, max_iter=50)
        r_newton = _score(c_newton)
        if r_newton > best_r:
            best_r = r_newton
            best_c = c_newton

    # Phase 5: Another SLSQP pass if time allows
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 12:
        c_polished2, r_polished2 = _slsqp_polish(best_c, best_r, t0, DEADLINE, max_iter=200)
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2

    # Phase 6: Final local search
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5:
        c_ls2, r_ls2 = _local_search(best_c, best_r, t0, DEADLINE, n_iterations=200)
        if r_ls2 > best_r:
            best_r = r_ls2
            best_c = c_ls2

    # Final validation
    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 2):
        best_c = _init_hexagonal()

    return best_c
