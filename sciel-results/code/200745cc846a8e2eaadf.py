"""Circle packing in a unit square: maximize common radius for 50 equal circles."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 150s kill

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c

def _soft_min_radius_and_grad(c, beta=50.0):
    """Compute soft-min radius and its gradient analytically (vectorized)."""
    n = len(c)
    
    # Pair distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances (per circle, min of x and y wall distances)
    wall_x = 0.5 - np.abs(c[:, 0])
    wall_y = 0.5 - np.abs(c[:, 1])
    wall_dists = np.minimum(wall_x, wall_y)
    
    # All distances: upper triangle pairs + wall distances
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min value
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    soft_min = min_val - np.log(sum_exp) / beta
    
    # Weights
    weights = exp_vals / sum_exp
    
    grad = np.zeros((n, 2))
    
    # Pair distance gradients (vectorized)
    pair_weights = weights[:len(pair_dists)]
    pair_diff = diff[iu]
    pair_d = dists[iu]
    
    safe_d = np.maximum(pair_d, 1e-12)
    directions = pair_diff / safe_d[:, np.newaxis]
    
    weighted_dirs = pair_weights[:, np.newaxis] * directions
    
    np.add.at(grad, iu[0], -weighted_dirs)
    np.add.at(grad, iu[1], weighted_dirs)
    
    # Wall distance gradients
    wall_weights = weights[len(pair_dists):]
    is_x_limit = wall_x <= wall_y
    x_sign = np.sign(c[:, 0])
    x_sign[x_sign == 0] = 1.0
    y_sign = np.sign(c[:, 1])
    y_sign[y_sign == 0] = 1.0
    
    grad[:, 0] += np.where(is_x_limit, wall_weights * x_sign, 0.0)
    grad[:, 1] += np.where(~is_x_limit, wall_weights * y_sign, 0.0)
    
    return soft_min, grad

def _init_grid_7x7():
    """7x7 grid with one extra circle at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])

def _init_hex(spacing=0.14):
    """Hexagonal lattice starting position."""
    positions = []
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions

def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))

def _init_clustered(rng):
    """Two clusters of 25 circles each."""
    c1 = rng.uniform(-0.4, -0.05, (25, 2))
    c2 = rng.uniform(0.05, 0.4, (25, 2))
    return np.vstack([c1, c2])

def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()

def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002, seed=999):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(seed)
    
    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)
        
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r

def _scipy_polish(centers, deadline=None):
    """Polish using scipy with smooth soft-min objective and analytical gradient."""
    if deadline is not None and time.time() > deadline - 10.0:
        return centers

    n = len(centers)
    x0 = centers.flatten()

    best_result = None
    best_r = _score(centers)

    for beta in [50.0, 100.0, 200.0, 400.0, 800.0]:
        if deadline is not None and time.time() > deadline - 5.0:
            break

        def objective(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            val, _ = _soft_min_radius_and_grad(c, beta)
            return -val

        def objective_grad(x):
            c = x.reshape(n, 2)
            c = np.clip(c, -0.5, 0.5)
            val, grad = _soft_min_radius_and_grad(c, beta)
            return -val, -grad

        bounds = [(-0.5, 0.5)] * (2 * n)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              jac=objective_grad,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            polished = result.x.reshape(n, 2)
            r = _score(polished)
            if r > best_r:
                best_r = r
                best_result = polished
                x0 = result.x.copy()
        except Exception:
            pass

    if best_result is not None and best_r >= _score(centers) - 1e-10:
        return best_result

    return centers

def _island_optimize(start_c, t0, deadline, seed, strategy="mixed"):
    """
    Run a complete optimization pipeline on a single island.
    Returns (best_centers, best_radius).
    """
    c = start_c.copy()
    best_c = c.copy()
    best_r = _score(c)
    
    if time.time() - t0 > deadline - 5:
        return best_c, best_r
    
    # Phase A: Grow radius via relaxation
    R_start = 0.05
    R_end = 0.072
    steps = 60
    for i in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        R = R_start + (R_end - R_start) * (i / steps)
        eta0 = 0.02 * (1 - i/steps) + 0.002
        eta1 = 0.002 * (1 - i/steps) + 0.0005
        c = _relax(c, R, 800, eta0, eta1, t0, deadline)
        r = _score(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if time.time() - t0 > deadline - 5:
        return best_c, best_r
    
    # Phase B: Langevin refinement
    if strategy in ("mixed", "langevin"):
        c, r = _langevin_refine(best_c, best_r, t0, deadline, 
                                steps=3000, T0=0.0012, T1=0.00005, 
                                lr=0.0025, seed=seed)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if time.time() - t0 > deadline - 8:
        return best_c, best_r
    
    # Phase C: Scipy polish
    if strategy in ("mixed", "polish"):
        polished = _scipy_polish(best_c, deadline)
        r = _score(polished)
        if r > best_r:
            best_r = r
            best_c = polished.copy()
    
    if time.time() - t0 > deadline - 4:
        return best_c, best_r
    
    # Phase D: Final fine relaxation
    r_cur = _score(best_c)
    c = _relax(best_c, r_cur, 1500, 0.005, 0.0003, t0, deadline)
    r = _score(c)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    return best_c, best_r

def construct():
    t0 = time.time()
    
    # Define multiple diverse islands with different strategies
    islands = []
    
    # Island 1: Grid 7x7 + mixed strategy
    islands.append((_init_grid_7x7(), 101, "mixed"))
    
    # Island 2: Hex lattice + mixed strategy
    islands.append((_init_hex(0.14), 202, "mixed"))
    
    # Island 3: Random + langevin-heavy
    rng3 = np.random.default_rng(303)
    islands.append((_init_rand(rng3), 303, "langevin"))
    
    # Island 4: Clustered + polish-heavy
    rng4 = np.random.default_rng(404)
    islands.append((_init_clustered(rng4), 404, "polish"))
    
    # Island 5: Grid 8x8 + mixed
    islands.append((_init_grid_8x8(), 505, "mixed"))
    
    # Island 6: Hex tighter + mixed
    islands.append((_init_hex(0.13), 606, "mixed"))
    
    # Island 7: Random different seed + mixed
    rng7 = np.random.default_rng(707)
    islands.append((_init_rand(rng7), 707, "mixed"))
    
    # Island 8: Clustered different seed + langevin
    rng8 = np.random.default_rng(808)
    islands.append((_init_clustered(rng8), 808, "langevin"))
    
    # Run islands in parallel (sequential but with time budgeting)
    n_islands = len(islands)
    results = []
    
    # Allocate time budget: give each island a fair share, but allow early ones to use more
    # if they finish quickly
    base_budget = (DEADLINE - (time.time() - t0) - 10) / n_islands
    
    for i, (start_c, seed, strategy) in enumerate(islands):
        if time.time() - t0 > DEADLINE - 8:
            break
        
        # Each island gets its own deadline
        island_deadline = min(DEADLINE - 5, t0 + base_budget * (i + 1) + 2)
        
        c, r = _island_optimize(start_c, t0, island_deadline, seed, strategy)
        results.append((c, r))
        
        # If we have time left, do a cross-island refinement
        if i < n_islands - 1 and time.time() - t0 < DEADLINE - 15:
            # Find current best
            best_r_so_far = max(r for _, r in results)
            # If this island did well, give it extra time for final polish
            if r > best_r_so_far - 0.0001:
                extra_deadline = min(DEADLINE - 3, island_deadline + 5)
                c2, r2 = _island_optimize(c, t0, extra_deadline, seed + 1000, "polish")
                if r2 > r:
                    results[i] = (c2, r2)
    
    # Select the best result
    if not results:
        best = _init_grid_7x7()
        return best
    
    best_c, best_r = max(results, key=lambda x: x[1])
    
    # Final global polish on the best result
    if time.time() - t0 < DEADLINE - 5:
        polished = _scipy_polish(best_c, DEADLINE)
        r = _score(polished)
        if r > best_r:
            best_c = polished
    
    # Final micro-relaxation
    if time.time() - t0 < DEADLINE - 2:
        r_cur = _score(best_c)
        c = _relax(best_c, r_cur, 1000, 0.003, 0.0002, t0, DEADLINE)
        r = _score(c)
        if r > _score(best_c):
            best_c = c
    
    # Safety check
    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 2):
        best_c = _init_grid_7x7()
    
    return best_c
