"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N_CIRCLES = 200
HALF_SIDE = 0.5


def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if centers is None:
        return 0.0
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0

    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=0.08)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)

    r = min(min_dist_to_wall, min_pair_dist / 2.0)
    return r


def hexagonal_initialization(seed=42):
    """Create a hexagonal lattice initialization with slight jitter for diversity."""
    rng = np.random.default_rng(seed)
    best_points = None
    best_score = 0.0

    for scale in [0.070, 0.072, 0.074, 0.076, 0.078, 0.080, 0.082]:
        dx = scale
        dy = scale * np.sqrt(3) / 2.0

        x_vals = np.arange(-0.5, 0.5 + dx / 2, dx)
        y_vals = np.arange(-0.5, 0.5 + dy / 2, dy)

        grid_points = []
        for i, y in enumerate(y_vals):
            for x in x_vals:
                if i % 2 == 1:
                    x_offset = dx / 2
                else:
                    x_offset = 0.0
                px = x + x_offset
                if -0.5 <= px <= 0.5 and -0.5 <= y <= 0.5:
                    grid_points.append([px, y])

        if len(grid_points) < N_CIRCLES:
            continue

        grid_points = np.array(grid_points)
        # Add small random jitter to break symmetry and avoid degenerate local minima
        jitter = rng.normal(0, 0.001, grid_points.shape)
        grid_points = grid_points + jitter
        grid_points = np.clip(grid_points, -0.5 + 1e-9, 0.5 - 1e-9)

        # Sort by distance from center and take N closest
        dists = np.sum(grid_points ** 2, axis=1)
        idx = np.argsort(dists)[:N_CIRCLES]
        points = grid_points[idx]

        r = compute_radius(points)
        if r > best_score:
            best_score = r
            best_points = points.copy()

    if best_points is not None:
        return best_points

    # Fallback: simple grid
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_CIRCLES]


def slsqp_solve(centers_init, r_init, deadline, max_iter=100):
    """
    Solve the hard-constrained NLP:
        maximize r
        subject to:
            ||c_i - c_j||^2 >= (2r)^2  for all i < j
            0.5 - |c_ik| >= r          for all i, k
            -0.5 <= c_ik <= 0.5
    Using SLSQP. Variable vector: [r, c_1x, c_1y, c_2x, c_2y, ...]
    """
    n = N_CIRCLES
    # Variable layout: x[0] = r, x[1:2n+1] = centers flattened
    x0 = np.zeros(1 + 2 * n)
    x0[0] = r_init
    x0[1:] = centers_init.flatten()

    # Precompute pair indices for constraints
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)

    def objective(x):
        # We maximize r, so minimize -r
        return -x[0]

    def constraints(x):
        r = x[0]
        c = x[1:].reshape(n, 2)

        cons = []

        # Pairwise non-overlap: ||c_i - c_j||^2 - (2r)^2 >= 0
        # Use vectorized computation for speed
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dist_sq = np.sum(diff ** 2, axis=1)
        # Constraint: dist_sq - 4*r^2 >= 0
        pair_cons = dist_sq - 4.0 * r * r
        # SLSQP expects array for each constraint function
        # We'll return as a single vectorized constraint
        cons.append(pair_cons)

        # Boundary constraints: 0.5 - |c_ik| >= r
        # For each coordinate:
        #   0.5 - c_x >= r  AND  c_x + 0.5 >= r  (i.e., -0.5 + r <= c_x <= 0.5 - r)
        # Similarly for y.
        # We express as: 0.5 - r - c_x >= 0 and c_x + 0.5 - r >= 0
        # Combine all four walls into one vector
        bx = c[:, 0]
        by = c[:, 1]
        wall_cons = np.concatenate([
            0.5 - r - bx,   # right wall
            bx + 0.5 - r,   # left wall
            0.5 - r - by,   # top wall
            by + 0.5 - r,   # bottom wall
        ])
        cons.append(wall_cons)

        return cons

    # Bounds: r >= 0, centers in [-0.5, 0.5]
    bounds = [(0.0, 0.5)] * (1 + 2 * n)

    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=[
                {'type': 'ineq', 'fun': lambda x: constraints(x)[0]},
                {'type': 'ineq', 'fun': lambda x: constraints(x)[1]},
            ],
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False},
        )
        if result.success or result.fun < 0:
            new_r = result.x[0]
            new_centers = result.x[1:].reshape(n, 2)
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            # Verify feasibility
            actual_r = compute_radius(new_centers)
            if actual_r > 0:
                return new_centers, actual_r
        return centers_init, r_init
    except Exception:
        return centers_init, r_init


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    np.random.seed(42)
    best_config = None
    best_r = 0.0

    # Multi-start SLSQP from diverse hexagonal initializations
    seeds = [42, 123, 456, 789, 1024, 2048]
    for s in seeds:
        if time.time() > deadline - 15.0:
            break
        centers = hexagonal_initialization(seed=s)
        r = compute_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers.copy()

    # Run SLSQP on the best initialization
    if best_config is not None and time.time() < deadline - 10.0:
        opt_centers, opt_r = slsqp_solve(best_config, best_r, deadline, max_iter=80)
        if opt_r > best_r:
            best_r = opt_r
            best_config = opt_centers

    # Second round: try SLSQP from a few other initializations
    for s in [123, 456, 789]:
        if time.time() > deadline - 10.0:
            break
        centers = hexagonal_initialization(seed=s)
        r = compute_radius(centers)
        if r < best_r:
            continue
        opt_centers, opt_r = slsqp_solve(centers, r, deadline, max_iter=60)
        if opt_r > best_r:
            best_r = opt_r
            best_config = opt_centers

    # Final refinement: one more SLSQP pass from the best solution
    if best_config is not None and time.time() < deadline - 5.0:
        opt_centers, opt_r = slsqp_solve(best_config, best_r, deadline, max_iter=50)
        if opt_r > best_r:
            best_r = opt_r
            best_config = opt_centers

    # Final validation
    final_r = compute_radius(best_config)
    if final_r <= 0 or best_config is None:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N_CIRCLES]
        final_r = compute_radius(best_config)

    return best_config
# EVOLVE-BLOCK-END
