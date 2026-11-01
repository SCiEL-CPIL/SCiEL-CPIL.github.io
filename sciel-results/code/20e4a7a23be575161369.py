"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)

    return min(min_pair_dist / 2.0, min_boundary)


def _get_distance_components(c):
    """Get all distance constraints (pairwise/2 and boundary) as a flat array."""
    n = c.shape[0]
    pair_dists = pdist(c, metric='euclidean') / 2.0
    bound_dists = 0.5 - np.abs(c)
    return np.concatenate([pair_dists, bound_dists.ravel()])


def _soft_min_objective(params, beta):
    """Minimize negative soft-min of distance components (log-sum-exp)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    dists = _get_distance_components(c)
    dists = np.maximum(dists, 1e-12)

    if beta > 5000:
        return -np.min(dists)

    d_min = np.min(dists)
    shifted = dists - d_min
    # Log-sum-exp trick: log(sum(exp(-beta*(d - d_min))))
    terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(terms))
    soft_min_val = d_min - log_sum / beta
    return -soft_min_val


def _soft_min_grad(params, beta):
    """Gradient of the soft-min objective (negative soft-min)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    dists = _get_distance_components(c)
    dists = np.maximum(dists, 1e-12)

    if beta > 5000:
        # Fall back to finite-difference-ish: just return zeros (L-BFGS-B will use num)
        return np.zeros_like(params)

    d_min = np.min(dists)
    shifted = dists - d_min
    terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(terms))
    # weights for soft-min gradient: w_i = exp(-beta*(d_i - d_min)) / sum
    w = terms / np.sum(terms)

    # d(soft_min)/d(c_j) = sum_i w_i * d(d_i)/d(c_j)
    # soft_min_obj = -soft_min, so grad_obj = -sum_i w_i * d(d_i)/d(c_j)

    grad = np.zeros_like(params)

    # Pairwise part: d_i = |c_a - c_b| / 2 for pair (a,b)
    # Contribution to grad for point k: sum over pairs involving k of w_i * (1/2) * (c_k - c_other)/|c_k - c_other|
    # pdist condensed index mapping: for pair (i,j) with i<j, index = i*n - i*(i+1)//2 + (j - i - 1)
    # We'll accumulate per-point.
    pair_dists = dists[:n*(n-1)//2]
    pair_w = w[:n*(n-1)//2]

    # For each point k, sum contributions from all other points
    # We need for each pair (i,j): unit vector (c_i - c_j)/|c_i - c_j|
    # Contribution to c_i: +w_ij * 0.5 * (c_i - c_j)/|c_i - c_j|
    # Contribution to c_j: -w_ij * 0.5 * (c_i - c_j)/|c_i - c_j|
    # But we need the index mapping. Let's loop over pairs for correctness (n=200 -> ~20k pairs, ok).
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = pair_w[idx]
            if w_ij > 1e-15:
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d > 1e-15:
                    unit = diff / d
                    # d(d_ij)/d(c_i) = 0.5 * unit ; d(d_ij)/d(c_j) = -0.5 * unit
                    # grad_obj = -sum w * d(d)/d(c)
                    grad[2*i:2*i+2] += -w_ij * 0.5 * unit
                    grad[2*j:2*j+2] += -w_ij * (-0.5 * unit)
            idx += 1

    # Boundary part: d = 0.5 - |c_k| for each coordinate
    # d(0.5 - |x|)/dx = -sign(x)
    bound_w = w[n*(n-1)//2:]
    for k in range(n):
        for coord in range(2):
            w_b = bound_w[k*2 + coord]
            if w_b > 1e-15:
                x = c[k, coord]
                s = np.sign(x)
                if s == 0:
                    s = 1.0
                # d(d_b)/d(x) = -s
                # grad_obj contribution = -w_b * (-s) = w_b * s
                grad[2*k + coord] += -w_b * (-s)

    return grad


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern optimized for the square."""
    best_c = None
    best_r = 0

    for d in np.linspace(0.055, 0.085, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2

        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1

        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    if best_c is not None:
        return best_c

    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_soft(c0, max_time=25.0):
    """Local optimization using L-BFGS-B on the soft-min surrogate with annealed beta."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)

    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)

    # Anneal beta from small to large for a smooth path
    betas = [10, 50, 200, 800, 3000]
    max_iters_per_beta = [150, 150, 200, 250, 300]

    for beta, max_iter in zip(betas, max_iters_per_beta):
        if time.time() > start_time + max_time - 2:
            break
        try:
            def obj(p):
                return _soft_min_objective(p, beta)

            def grad(p):
                return _soft_min_grad(p, beta)

            result = minimize(
                obj,
                best_c.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )

            if result.fun < -best_r:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass

    return best_c, best_r


def _final_polish(c0, max_time=15.0):
    """High-precision polish: very high beta soft-min or direct L-BFGS-B on raw min."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    bounds = [(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N)

    # Use a very high beta soft-min for near-exact min
    for beta in [10000, 50000]:
        if time.time() > start_time + max_time - 1:
            break
        try:
            def obj(p):
                return _soft_min_objective(p, beta)

            def grad(p):
                return _soft_min_grad(p, beta)

            result = minimize(
                obj,
                best_c.flatten(),
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )

            if result.fun < -best_r:
                c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_new = _compute_radius(c_new)
                if r_new > best_r:
                    best_c = c_new
                    best_r = r_new
        except Exception:
            pass

    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    best_radius = 0.0
    best_c = None

    # Primary: hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # Optimize the hexagonal init with soft-min
    if time.time() < deadline - 25:
        c_opt, r_opt = _local_optimize_soft(c_hex, max_time=min(35.0, deadline - time.time() - 20))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # Multi-start with random inits
    np.random.seed(123)
    for trial in range(4):
        if time.time() > deadline - 30:
            break

        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        r_rand = _compute_radius(c_rand)

        if r_rand > best_radius:
            best_radius = r_rand
            best_c = c_rand.copy()

        if time.time() < deadline - 25:
            remaining = deadline - time.time() - 20
            if remaining > 15:
                c_opt, r_opt = _local_optimize_soft(c_rand, max_time=min(25.0, remaining))
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_c = c_opt.copy()

    # Final high-precision polish on the best solution
    if time.time() < deadline - 10 and best_c is not None:
        c_polished, r_polished = _final_polish(best_c, max_time=min(15.0, deadline - time.time() - 5))
        if r_polished > best_radius:
            best_radius = r_polished
            best_c = c_polished.copy()

    if best_c is None:
        best_c = _initialize_hexagonal()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
# EVOLVE-BLOCK-END
