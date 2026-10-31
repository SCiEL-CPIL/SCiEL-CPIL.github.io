"""Circle packing in unit square: 26 disjoint circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26


def _objective(z):
    """Negative sum of radii plus penalties for constraint violations."""
    n = len(z) // 3
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    score = -np.sum(r)
    # Wall violations
    score += 50.0 * np.sum(np.maximum(0, r - x) ** 2)
    score += 50.0 * np.sum(np.maximum(0, x + r - 1) ** 2)
    score += 50.0 * np.sum(np.maximum(0, r - y) ** 2)
    score += 50.0 * np.sum(np.maximum(0, y + r - 1) ** 2)
    # Pairwise overlap penalties
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    rsum = r[:, None] + r[None, :]
    overlap = np.maximum(0, rsum - dist)
    iu = np.triu_indices(n, k=1)
    score += 100.0 * np.sum(overlap[iu] ** 2)
    return score


def _is_valid(xs, ys, r, tol=1e-7):
    """Check all constraints are satisfied."""
    n = len(xs)
    # Wall checks
    if np.any(xs + r > 1 + tol) or np.any(xs - r < -tol):
        return False
    if np.any(ys + r > 1 + tol) or np.any(ys - r < -tol):
        return False
    # Pairwise
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    rsum = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < rsum[iu] - tol):
        return False
    return True


def _post_process(xs, ys, r):
    """Clamp to feasible set."""
    r = np.clip(r, 1e-4, 0.5)
    xs = np.clip(xs, r, 1 - r)
    ys = np.clip(ys, r, 1 - r)
    return xs, ys, r


def _initial_corner_heavy(seed):
    """Place circles with emphasis on corners and edges for larger radii."""
    rng = np.random.default_rng(seed)
    # Pre-defined corner and edge positions
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    edges = [(0.5, 0.12), (0.12, 0.5), (0.5, 0.88), (0.88, 0.5),
             (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
    positions = corners + edges
    # Fill remaining with random
    while len(positions) < N:
        positions.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))
    positions = positions[:N]
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    r = np.full(N, 0.05)
    return xs, ys, r


def _initial_lattice(seed):
    """Grid initialization with jitter."""
    rng = np.random.default_rng(seed)
    side = 6
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            positions.append(((i + 0.5) / side, (j + 0.5) / side))
    positions = positions[:N]
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    xs += rng.uniform(-0.02, 0.02, size=N)
    ys += rng.uniform(-0.02, 0.02, size=N)
    r = np.full(N, 0.05)
    return xs, ys, r


def _initial_random(seed):
    """Pure random initialization."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1, 0.9, size=N)
    ys = rng.uniform(0.1, 0.9, size=N)
    r = np.full(N, 0.05)
    return xs, ys, r


def _force_directed_anneal(xs, ys, r, deadline):
    """Force-directed with simulated annealing to escape local minima."""
    n = N
    current_temp = 1.0
    cooling = 0.999
    min_temp = 0.001
    best_score = np.sum(r)
    best_xs, best_ys, best_r = xs.copy(), ys.copy(), r.copy()
    
    lr = 0.01
    for step in range(30000):
        if time.time() > deadline:
            break
        
        # Compute forces
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, 1e-8)
        rsum = r[:, None] + r[None, :]
        overlap = np.maximum(0, rsum - dist)
        ux = dx / dist
        uy = dy / dist
        
        force_x = 2.0 * np.sum(overlap * ux, axis=1)
        force_y = 2.0 * np.sum(overlap * uy, axis=1)
        
        # Wall forces
        left_viol = np.maximum(0, r - xs)
        right_viol = np.maximum(0, xs + r - 1)
        bottom_viol = np.maximum(0, r - ys)
        top_viol = np.maximum(0, ys + r - 1)
        force_x += 100.0 * (left_viol - right_viol)
        force_y += 100.0 * (bottom_viol - top_viol)
        
        # Radius gradient
        overlap_rad = np.sum(overlap, axis=1)
        wall_rad = left_viol + right_viol + bottom_viol + top_viol
        grad_r = 1.0 - 10.0 * overlap_rad - 50.0 * wall_rad
        
        # Update with noise
        noise = np.sqrt(current_temp) * 0.005
        xs += lr * force_x + rng_uniform_noise(noise, n, step)
        ys += lr * force_y + rng_uniform_noise(noise, n, step + 1000)
        r += lr * grad_r + rng_uniform_noise(noise * 0.5, n, step + 2000)
        
        r = np.clip(r, 0.01, 0.45)
        xs, ys = _post_process(xs, ys, r)
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_r = xs.copy(), ys.copy(), r.copy()
        
        current_temp = max(min_temp, current_temp * cooling)
    
    return best_xs, best_ys, best_r


def rng_uniform_noise(scale, n, seed):
    """Quick deterministic noise."""
    if scale < 1e-8:
        return np.zeros(n)
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=n)


def _slsqp_polish(xs, ys, r, deadline):
    """Polish with SLSQP for tight constraint handling."""
    z = np.concatenate([xs, ys, r])
    n = N
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-4, 0.5))  # x
    for i in range(n):
        bounds.append((1e-4, 0.5))  # y
    for i in range(n):
        bounds.append((1e-4, 0.45))  # r
    
    constraints = []
    # Wall constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[i] - z[2*n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[n + i] - z[2*n + i]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: 1 - z[n + i] - z[2*n + i]})
    
    result = minimize(_objective, z, method='SLSQP', bounds=bounds, 
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
    
    if result.success and result.fun < _objective(z):
        z_opt = result.x
        new_xs = z_opt[0::3]
        new_ys = z_opt[1::3]
        new_r = z_opt[2::3]
        if _is_valid(new_xs, new_ys, new_r):
            return new_xs, new_ys, new_r
    return xs, ys, r


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    best_score = -np.inf
    best_xs = best_ys = best_r = None
    
    initializers = [_initial_corner_heavy, _initial_lattice, _initial_random]
    init_idx = 0
    
    for seed in range(100):
        if time.time() > deadline - 3.0:
            break
        
        # Cycle through initializers
        init_func = initializers[init_idx % len(initializers)]
        init_idx += 1
        
        xs, ys, r = init_func(seed)
        xs, ys, r = _post_process(xs, ys, r)
        
        # Force-directed anneal
        xs, ys, r = _force_directed_anneal(xs, ys, r, deadline - 10.0)
        
        # SLSQP polish if time permits
        if time.time() < deadline - 5.0:
            xs, ys, r = _slsqp_polish(xs, ys, r, deadline - 5.0)
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_r = xs.copy(), ys.copy(), r.copy()
        
        # Early termination if we're doing well
        if best_score > 0.75 and time.time() > start_time + 60.0:
            break
    
    if best_xs is None:
        side = 6
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.001)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack([best_xs, best_ys, best_r])
