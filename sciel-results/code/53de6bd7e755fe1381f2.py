"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0 - 10.0  # safety margin before the 150s kill

def _radius(centers):
    """Vectorized max valid radius. Returns 0.0 if invalid."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # boundary distance
    b = 0.5 - np.abs(centers)
    min_b = float(np.min(b))
    if min_b <= 0.0:
        return 0.0
    # pairwise distances via broadcasting
    diff = centers[:, None, :] - centers[None, :, :]
    d_sq = np.sum(diff * diff, axis=2)
    # exclude self
    np.fill_diagonal(d_sq, np.inf)
    min_d_sq = float(np.min(d_sq))
    if min_d_sq <= 0.0:
        return 0.0
    min_p = np.sqrt(min_d_sq)
    return min(min_p / 2.0, min_b)

def _soft_min(x, beta=100.0):
    """Approximate minimum using log-sum-exp."""
    m = np.min(x)
    if not np.isfinite(m):
        return m
    return m - np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _smooth_radius(centers, beta=50.0):
    """Smooth surrogate for radius optimization."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    # boundary distance
    b = 0.5 - np.abs(centers)
    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    d_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d_sq, np.inf)
    d = np.sqrt(d_sq)
    d_half = d / 2.0
    # Vector of boundary distances
    b_flat = b.flatten()
    # Vector of half pairwise distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_flat = d_half[i_idx, j_idx]
    all_constraints = np.concatenate([b_flat, d_flat])
    return _soft_min(all_constraints, beta=beta)

def _neg_smooth_radius(params, beta=50.0):
    """Objective for minimize: negative smooth radius (maximize)."""
    return -_smooth_radius(params.reshape(N, 3), beta=beta)

def _cubic_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _hexagonal_layers():
    """Create a hexagonal close-packed arrangement in layers."""
    # For 50 spheres, try 5 layers of 10 spheres each
    # Hexagonal packing in 2D: 10 spheres in a hexagon-like pattern
    # Actually, let's use a grid with hexagonal offset
    centers = []
    # 5 layers
    for layer in range(5):
        z = -0.5 + (layer + 0.5) / 5
        # In each layer, place 10 spheres
        # Use a 5x2 grid with hexagonal offset
        for i in range(5):
            for j in range(2):
                x = -0.5 + (i + 0.5) / 5
                y = -0.5 + (j + 0.5) / 2
                # Add hexagonal offset for odd rows
                if j % 2 == 1:
                    x += 0.5 / 5
                centers.append([x, y, z])
    return np.array(centers[:N])

def _random_init(rng):
    margin = 0.1
    return rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))

def _local_optimize(centers, budget, beta_schedule=None):
    """Run L-BFGS-B from a start; return best centers and radius."""
    t0 = time.time()
    params = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    best_params = params.copy()
    best_r = _radius(centers)
    
    if beta_schedule is None:
        beta_schedule = [50.0, 100.0, 200.0, 500.0]
    
    for beta in beta_schedule:
        if time.time() - t0 > budget:
            break
        try:
            result = minimize(
                lambda p: _neg_smooth_radius(p, beta),
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100,
                         'maxfun': 10000},
            )
            new_c = result.x.reshape(N, 3)
            new_r = _radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
            params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_r, time.time() - t0

def _anneal(centers, budget, rng):
    """Simulated annealing to escape local minima."""
    t0 = time.time()
    centers = centers.copy()
    cur_r = _radius(centers)
    best = centers.copy()
    best_r = cur_r
    T = 0.02
    cool = 0.99
    while time.time() - t0 < budget:
        if T < 1e-7:
            break
        idx = rng.integers(N)
        delta = rng.normal(0, T * 0.15, size=3)
        new_c = centers.copy()
        new_c[idx] += delta
        new_c[idx] = np.clip(new_c[idx], -0.5, 0.5)
        new_r = _radius(new_c)
        dr = new_r - cur_r
        if dr > 0 or rng.random() < np.exp(dr / T):
            centers = new_c
            cur_r = new_r
            if cur_r > best_r:
                best_r = cur_r
                best = new_c.copy()
        T *= cool
    return best, best_r, time.time() - t0

def _polish(centers, budget):
    """High-precision polish pass with tighter tolerances."""
    t0 = time.time()
    params = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    best_params = params.copy()
    best_r = _radius(centers)
    
    # Use very high beta for near-exact optimization
    for beta in [500.0, 1000.0, 2000.0]:
        if time.time() - t0 > budget:
            break
        try:
            result = minimize(
                lambda p: _neg_smooth_radius(p, beta),
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 200,
                         'maxfun': 20000},
            )
            new_c = result.x.reshape(N, 3)
            new_r = _radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
            params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_r, time.time() - t0

def construct():
    global_start = time.time()
    rng = np.random.default_rng(12345)

    # Multiple initial configurations
    best_centers = _cubic_grid()
    best_r = _radius(best_centers)
    
    # Try hexagonal layers
    hex_centers = _hexagonal_layers()
    hex_r = _radius(hex_centers)
    if hex_r > best_r:
        best_r = hex_r
        best_centers = hex_centers

    def remaining():
        return DEADLINE - (time.time() - global_start)

    # 1) Optimize from cubic grid
    if remaining() > 25.0:
        c, r, _ = _local_optimize(best_centers, min(30.0, remaining() - 15.0))
        if r > best_r:
            best_r, best_centers = r, c

    # 2) Optimize from hexagonal layers
    if remaining() > 25.0:
        c, r, _ = _local_optimize(hex_centers, min(30.0, remaining() - 15.0))
        if r > best_r:
            best_r, best_centers = r, c

    # 3) Several random restarts with optimization
    for i in range(8):
        if remaining() < 30.0:
            break
        c0 = _random_init(rng)
        c, r, _ = _local_optimize(c0, min(25.0, remaining() - 20.0))
        if r > best_r:
            best_r, best_centers = r, c

    # 4) Anneal the best, then re-optimize
    if remaining() > 35.0:
        c, r, _ = _anneal(best_centers, min(30.0, remaining() - 25.0), rng)
        if r > best_r:
            best_r, best_centers = r, c
    if remaining() > 20.0:
        c, r, _ = _local_optimize(best_centers, min(15.0, remaining() - 12.0))
        if r > best_r:
            best_r, best_centers = r, c

    # 5) Final high-precision polish
    if remaining() > 15.0:
        c, r, _ = _polish(best_centers, min(12.0, remaining() - 10.0))
        if r > best_r:
            best_r, best_centers = r, c

    # final validity guard
    final = np.clip(best_centers, -0.5, 0.5)
    fr = _radius(final)
    if fr <= 0.0:
        final = _cubic_grid()
    return final
# EVOLVE-BLOCK-END
