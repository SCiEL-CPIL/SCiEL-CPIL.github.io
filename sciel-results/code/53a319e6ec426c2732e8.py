"""Improved sphere packing optimizer for 100 equal spheres in a unit cube."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = 140.0  # safety margin before 150s kill


def _compute_radius(c):
    """Compute the achievable common radius for centres c (N,3)."""
    n = c.shape[0]
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min(min_dist / 2.0, min_wall)


def _make_grid(nx, ny, nz, jitter=0.0, rng=None):
    """Create a grid of points in the unit cube centered at origin."""
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        if rng is None:
            rng = np.random.RandomState(0)
        grid = grid + rng.uniform(-jitter, jitter, grid.shape)
    return grid


def _make_bcc_like(n, rng):
    """Create a BCC-like arrangement for n spheres."""
    side = int(np.ceil(n ** (1.0 / 3.0)))
    while side ** 3 < n:
        side += 1
    # Body-centered cubic: corners + center of each cell
    xs = (np.arange(side) + 0.5) / side - 0.5
    ys = (np.arange(side) + 0.5) / side - 0.5
    zs = (np.arange(side) + 0.5) / side - 0.5
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append((x, y, z))
                # Add body center offset
                if side > 1:
                    points.append((x + 0.5 / side, y + 0.5 / side, z + 0.5 / side))
    points = np.array(points)
    if len(points) > n:
        center_dist = np.sum(points * points, axis=1)
        idx = np.argsort(center_dist)[:n]
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        extra = rng.uniform(-0.4, 0.4, (n - len(points), 3))
        points = np.vstack([points, extra])
    return points


def _make_fcc_like(n, rng):
    """Create an FCC-like arrangement for n spheres."""
    side = int(np.ceil(n ** (1.0 / 3.0)))
    while side ** 3 < n:
        side += 1
    grid = _make_grid(side, side, side, jitter=0.01, rng=rng)
    center_dist = np.sum(grid * grid, axis=1)
    idx = np.argsort(center_dist)[:n]
    return grid[idx]


def _relax_spheres(c, r, max_iter=5000, tol=1e-10):
    """
    Relax spheres to satisfy radius r using vectorized repulsion.
    Returns (c_new, converged)
    """
    c = c.copy()
    n = c.shape[0]
    limit = 0.5 - r
    lr = 0.15
    two_r = 2.0 * r

    for it in range(max_iter):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        dist = np.sqrt(dist_sq)

        dist_safe = np.where(dist > 1e-12, dist, 1.0)
        
        # Overlap force: strong repulsion when overlapping
        overlap = np.where(dist < two_r, two_r - dist, 0.0)
        # Use inverse-distance weighted force
        force_mag = np.where(dist < two_r, overlap / dist_safe, 0.0)
        # Add a small constant repulsion for very close spheres
        very_close = np.where(dist < 0.5 * two_r, 1.0, 0.0)
        force_mag += very_close / dist_safe * 0.1

        force = np.sum(diff * force_mag[:, :, np.newaxis], axis=1)

        c += lr * force
        c = np.clip(c, -limit, limit)

        max_force = np.max(np.abs(force))
        if max_force < tol:
            return c, True

        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.97

    return c, False


def _soft_min_radius(c, beta=30.0):
    """Differentiable soft-min approximation of the packing radius."""
    n = c.shape[0]
    wall = 0.5 - np.max(np.abs(c), axis=1)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    half_dists_sq = dist_sq / 4.0
    
    mask = ~np.eye(n, dtype=bool)
    valid_half_dists_sq = np.where(mask, half_dists_sq, np.inf)
    
    min_half_sq = np.min(valid_half_dists_sq)
    exp_terms = np.exp(-beta * (valid_half_dists_sq - min_half_sq))
    soft_min_half_sq = min_half_sq - np.log(np.sum(exp_terms)) / beta
    
    min_wall = np.min(wall)
    exp_wall = np.exp(-beta * (wall - min_wall))
    soft_min_wall = min_wall - np.log(np.sum(exp_wall)) / beta
    
    vals = np.array([soft_min_half_sq, soft_min_wall])
    min_val = np.min(vals)
    exp_all = np.exp(-beta * (vals - min_val))
    combined = min_val - np.log(np.sum(exp_all)) / beta
    
    return np.sqrt(max(combined, 1e-12))


def _polish_with_scipy(c, r_target, time_budget=8.0, beta=35.0, maxiter=200):
    """Use scipy L-BFGS-B to polish the configuration."""
    t0 = time.time()
    n = c.shape[0]
    c_flat = c.flatten().copy()
    limit = 0.5 - r_target

    def objective(x):
        if time.time() - t0 > time_budget:
            return 1e10
        c_x = x.reshape(n, 3)
        outside = np.sum(np.maximum(0, np.abs(c_x) - limit)) ** 2
        r_soft = _soft_min_radius(c_x, beta=beta)
        return -r_soft + 50.0 * outside

    def gradient(x):
        if time.time() - t0 > time_budget:
            return np.zeros_like(x)
        c_x = x.reshape(n, 3)
        
        # Numerical gradient (simpler and more robust)
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        
        return grad

    try:
        result = minimize(
            objective, c_flat, jac=gradient,
            method='L-BFGS-B',
            bounds=[(-limit, limit)] * (n * 3),
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 15}
        )
        c_new = result.x.reshape(n, 3)
        c_new = np.clip(c_new, -limit, limit)
        return c_new
    except Exception:
        return c


def _local_search(c, best_r, rng, time_budget=10.0):
    """Local search with random perturbations and relaxation."""
    t0 = time.time()
    c_best = c.copy()
    r_best = best_r
    
    for trial in range(50):
        if time.time() - t0 > time_budget:
            break
        
        # Random perturbation
        scale = 0.003 * (1.0 + 0.5 * np.random.rand())
        c_try = c_best + rng.randn(*c_best.shape) * scale
        c_try = np.clip(c_try, -(0.5 - r_best), 0.5 - r_best)
        
        # Relax
        c_relaxed, _ = _relax_spheres(c_try, r_best, max_iter=1000)
        r_try = _compute_radius(c_relaxed)
        
        if r_try > r_best:
            r_best = r_try
            c_best = c_relaxed.copy()
    
    return c_best, r_best


def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)
    
    best_c = None
    best_r = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    # Grid configurations
    grid_configs = [(5, 5, 4), (4, 5, 5), (5, 4, 5), (4, 4, 7), (4, 7, 4), (7, 4, 4)]
    for nx, ny, nz in grid_configs:
        for jitter in [0.0, 0.003, 0.008]:
            c0 = _make_grid(nx, ny, nz, jitter=jitter, rng=rng)
            if len(c0) >= N:
                c0 = c0[:N]
            elif len(c0) < N:
                extra = rng.uniform(-0.4, 0.4, (N - len(c0), 3))
                c0 = np.vstack([c0, extra])
            c0 = np.clip(c0, -0.48, 0.48)
            inits.append(c0)
    
    # FCC-like
    for seed in [42, 123, 456]:
        rng_local = np.random.RandomState(seed)
        c0 = _make_fcc_like(N, rng_local)
        c0 = np.clip(c0, -0.48, 0.48)
        inits.append(c0)
    
    # BCC-like
    for seed in [42, 789, 101]:
        rng_local = np.random.RandomState(seed)
        c0 = _make_bcc_like(N, rng_local)
        c0 = np.clip(c0, -0.48, 0.48)
        inits.append(c0)
    
    # Random uniform
    for seed in [42, 2024]:
        rng_local = np.random.RandomState(seed)
        c0 = rng_local.uniform(-0.45, 0.45, (N, 3))
        inits.append(c0)
    
    # Process each initial configuration
    for c0 in inits:
        if time.time() - t0 > _DEADLINE - 25:
            break
        
        r_current = _compute_radius(c0)
        if r_current > best_r:
            best_r = r_current
            best_c = c0.copy()
        
        # Incremental radius increase with relaxation
        r_target = r_current * 1.003
        max_r_target = 0.113
        fail_count = 0
        
        while time.time() - t0 < _DEADLINE - 25 and r_target <= max_r_target:
            c_test, converged = _relax_spheres(c0, r_target, max_iter=3000)
            r_actual = _compute_radius(c_test)
            
            if r_actual >= r_target - 1e-9 and r_actual > best_r:
                best_r = r_actual
                best_c = c_test.copy()
                c0 = c_test.copy()
                r_target *= 1.003
                fail_count = 0
            elif r_actual > best_r:
                best_r = r_actual
                best_c = c_test.copy()
                c0 = c_test.copy()
                r_target = r_actual * 1.002
                fail_count = 0
            else:
                fail_count += 1
                if fail_count >= 4:
                    break
                r_target = (r_target + best_r) / 2.0
                if r_target <= best_r + 1e-10:
                    break
    
    # Local search refinement
    if best_c is not None and time.time() - t0 < _DEADLINE - 20:
        c_ls, r_ls = _local_search(best_c, best_r, rng, time_budget=min(15.0, _DEADLINE - (time.time() - t0) - 10))
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    # Scipy polish
    if best_c is not None and time.time() - t0 < _DEADLINE - 12:
        remaining = _DEADLINE - (time.time() - t0)
        c_polished = _polish_with_scipy(best_c, best_r * 1.0005, time_budget=min(8.0, remaining - 6.0), beta=30.0, maxiter=150)
        r_polished = _compute_radius(c_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Second scipy polish with higher beta
    if best_c is not None and time.time() - t0 < _DEADLINE - 6:
        remaining = _DEADLINE - (time.time() - t0)
        c_polished2 = _polish_with_scipy(best_c, best_r * 1.0002, time_budget=min(5.0, remaining - 3.0), beta=50.0, maxiter=200)
        r_polished2 = _compute_radius(c_polished2)
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()
    
    # Final local search
    if best_c is not None and time.time() - t0 < _DEADLINE - 3:
        c_final, r_final = _local_search(best_c, best_r, rng, time_budget=min(5.0, _DEADLINE - (time.time() - t0) - 1.0))
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    # Ensure valid output
    if best_c is None:
        best_c = _make_grid(5, 5, 4)[:N]
    
    # Final safety clip
    min_r = _compute_radius(best_c)
    limit = 0.5 - min_r
    best_c = np.clip(best_c, -limit, limit)
    
    return best_c
