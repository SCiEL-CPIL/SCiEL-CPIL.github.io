"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _min_radius(x):
    """Compute the true minimum radius for configuration x (N,3)."""
    x = x.reshape(N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    wall_dists = np.min(np.abs(0.5 - np.abs(x)), axis=1)
    min_wall_dist = np.min(wall_dists)

    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_radius(x, beta):
    """
    Smooth approximation of -min_radius using log-sum-exp soft-min.
    Returns a scalar: the soft-min of all constraint values (pair distances/2 and wall distances).
    We minimize this (equivalent to maximizing the soft-min of the radii).
    """
    x = x.reshape(N, 3)
    
    # Pair distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e-30)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(x), axis=1)
    
    # All constraint values (we want to maximize the min of these)
    # Pair constraints: dists[i,j]/2 for i<j
    # Wall constraints: wall_dists[i]
    
    # Get upper triangle indices for pairs
    iu = np.triu_indices(N, 1)
    pair_vals = dists[iu] / 2.0
    wall_vals = wall_dists
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft min: min(x) ≈ -(1/beta) * log(sum(exp(-beta*x)))
    # Equivalently, soft_min(x) = -soft_max(-x)
    # soft_max(x) = (1/beta) * log(sum(exp(beta*x)))
    
    # For numerical stability, subtract max
    neg_vals = -all_vals
    v_max = np.max(neg_vals)
    exp_vals = np.exp(beta * (neg_vals - v_max))
    soft_max_neg = v_max + (1.0 / beta) * np.log(np.sum(exp_vals))
    
    # soft_min(all_vals) = -soft_max(-all_vals)
    return -soft_max_neg


def _gradient_soft_min_radius(x, beta):
    """
    Analytical gradient of _soft_min_radius with respect to x.
    
    soft_min(x_1, ..., x_m) = -(1/beta) * log(sum_k exp(-beta * x_k))
    
    d(soft_min)/dx_i = (1/beta) * sum_k (exp(-beta*x_k) / sum_j exp(-beta*x_j)) * (d(-x_k)/dx_i)
                    = -sum_k w_k * (dx_k/dx_i)
    where w_k = exp(-beta*x_k) / sum_j exp(-beta*x_j)
    """
    x = x.reshape(N, 3)
    
    # Pair distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e-30)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(x), axis=1)
    
    # Get upper triangle indices for pairs
    iu = np.triu_indices(N, 1)
    pair_vals = dists[iu] / 2.0
    wall_vals = wall_dists
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Compute weights w_k = exp(-beta*x_k) / sum_j exp(-beta*x_j)
    neg_vals = -all_vals
    v_max = np.max(neg_vals)
    exp_vals = np.exp(beta * (neg_vals - v_max))
    w = exp_vals / np.sum(exp_vals)
    
    # Now compute gradient
    grad = np.zeros((N, 3))
    
    # Pair contributions
    # For pair (i,j): x_k = dists[i,j]/2
    # d(x_k)/dx_i = (x_i - x_j) / dists[i,j]
    # d(x_k)/dx_j = (x_j - x_i) / dists[i,j]
    
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-15:
            continue
        w_k = w[idx]
        direction = (x[i] - x[j]) / d
        grad[i] -= w_k * direction
        grad[j] += w_k * direction
    
    # Wall contributions
    # For sphere i: x_k = min over axis a of (0.5 - |x_ia|)
    # The gradient depends on which axis is the minimum
    n_pairs = len(iu[0])
    for i in range(N):
        w_k = w[n_pairs + i]
        # Find which axis gives the minimum wall distance
        abs_x = np.abs(x[i])
        wall_per_axis = 0.5 - abs_x
        min_axis = np.argmin(wall_per_axis)
        # d(0.5 - |x_ia|)/dx_ia = -sign(x_ia)
        sign_val = np.sign(x[i, min_axis])
        if sign_val == 0:
            sign_val = 1.0
        grad[i, min_axis] -= w_k * sign_val
    
    return grad


def _fibonacci_lattice_init(N, seed=42):
    """Generate initial points using Fibonacci lattice for good distribution."""
    rng = np.random.default_rng(seed)
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = np.zeros((N, 3))
    
    for i in range(N):
        t = i / N
        z = 2 * t - 1
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    
    points *= 0.45
    points += rng.normal(0, 0.01, size=(N, 3))
    points = np.clip(points, -0.48, 0.48)
    
    return points


def _grid_init(N):
    """Grid-based initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init(N, seed=123):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _best_first_grid_init(N):
    """
    Best-first grid initialization: place spheres at grid positions 
    that maximize the minimum distance to already-placed spheres.
    """
    n_per_dim = 8
    ticks = (np.arange(n_per_dim) + 0.5) / n_per_dim - 0.5
    candidates = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    placed = []
    for _ in range(N):
        if len(candidates) == 0:
            break
        
        if len(placed) == 0:
            best_idx = np.argmin(np.sum(candidates ** 2, axis=1))
            placed.append(candidates[best_idx].copy())
            candidates = np.delete(candidates, best_idx, axis=0)
        else:
            min_dists = np.min(
                np.sqrt(np.sum((candidates[:, np.newaxis, :] - np.array(placed)[np.newaxis, :, :]) ** 2, axis=2)),
                axis=1
            )
            wall_dists = np.min(np.abs(0.5 - np.abs(candidates)), axis=1)
            scores = np.minimum(min_dists, wall_dists)
            best_idx = np.argmax(scores)
            placed.append(candidates[best_idx].copy())
            candidates = np.delete(candidates, best_idx, axis=0)
    
    while len(placed) < N:
        rng = np.random.default_rng(len(placed))
        placed.append(rng.uniform(-0.45, 0.45, size=3))
    
    return np.array(placed[:N])


def _polish(x_init, max_iter=2000, time_limit=30.0):
    """
    High-precision polish: focus on active constraints with smaller steps.
    """
    x = x_init.copy().reshape(N, 3)
    x = np.clip(x, -0.49, 0.49)
    
    best_x = x.copy()
    best_score = _min_radius(x)
    
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.min(0.5 - np.abs(x), axis=1)
        
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        threshold = current_r * 1.1
        
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2.0 * threshold:
                    direction = x[i] - x[j]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-15:
                        continue
                    direction /= norm
                    step = 0.002
                    x_new = x.copy()
                    x_new[i] += step * direction
                    x_new[j] -= step * direction
                    x_new = np.clip(x_new, -0.49, 0.49)
                    new_score = _min_radius(x_new)
                    if new_score > current_r + 1e-14:
                        x = x_new
                        moved = True
        
        for i in range(N):
            if wall_dists[i] < threshold:
                direction = -x[i]
                norm = np.linalg.norm(direction)
                if norm < 1e-15:
                    continue
                direction /= norm
                step = 0.002
                x_new = x.copy()
                x_new[i] += step * direction
                x_new = np.clip(x_new, -0.49, 0.49)
                new_score = _min_radius(x_new)
                if new_score > current_r + 1e-14:
                    x = x_new
                    moved = True
        
        if not moved:
            for i in range(N):
                for j in range(i + 1, N):
                    d = dists[i, j]
                    if d < 2.0 * threshold:
                        direction = x[i] - x[j]
                        norm = np.linalg.norm(direction)
                        if norm < 1e-15:
                            continue
                        direction /= norm
                        step = 0.0005
                        x_new = x.copy()
                        x_new[i] += step * direction
                        x_new[j] -= step * direction
                        x_new = np.clip(x_new, -0.49, 0.49)
                        new_score = _min_radius(x_new)
                        if new_score > current_r + 1e-15:
                            x = x_new
                            moved = True
            
            for i in range(N):
                if wall_dists[i] < threshold:
                    direction = -x[i]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-15:
                        continue
                    direction /= norm
                    step = 0.0005
                    x_new = x.copy()
                    x_new[i] += step * direction
                    x_new = np.clip(x_new, -0.49, 0.49)
                    new_score = _min_radius(x_new)
                    if new_score > current_r + 1e-15:
                        x = x_new
                        moved = True
            
            if not moved:
                break
        
        current_score = _min_radius(x)
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
    
    return best_x, best_score


def _scipy_refine_annealed(x_init, time_limit=20.0):
    """
    Use scipy minimize with soft-min objective and beta annealing for refinement.
    Start with low beta (smooth) and increase to high beta (sharp) to navigate landscape.
    """
    t0 = time.time()
    x = x_init.copy().reshape(N, 3)
    
    # Beta annealing schedule
    beta_schedule = [20, 50, 100, 200, 400, 800, 1200]
    
    for beta in beta_schedule:
        if time.time() - t0 > time_limit:
            break
        
        def objective(x_flat):
            return _soft_min_radius(x_flat, beta)
        
        def gradient(x_flat):
            return _gradient_soft_min_radius(x_flat, beta)
        
        try:
            result = minimize(
                objective,
                x,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            x = np.clip(result.x.reshape(N, 3), -0.49, 0.49)
        except Exception:
            break
    
    refined_x = np.clip(x, -0.5, 0.5)
    score = _min_radius(refined_x)
    return refined_x, score


def _repulsion_optimize(x_init, max_iter=5000, time_limit=100.0, verbose=False):
    """
    Repulsion-based optimization for sphere packing.
    """
    x = x_init.copy().reshape(N, 3)
    x = np.clip(x, -0.49, 0.49)
    
    best_x = x.copy()
    best_score = _min_radius(x)
    
    t0 = time.time()
    
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.min(0.5 - np.abs(x), axis=1)
        
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx[0], min_pair_idx[1]
        
        wall_idx = np.argmin(wall_dists)
        
        if min_pair_dist / 2.0 <= min_wall_dist:
            direction = x[i] - x[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = np.random.randn(3)
                norm = np.linalg.norm(direction)
            direction /= norm
            
            step = 0.01
            for _ in range(20):
                x_new = x.copy()
                x_new[i] += step * direction
                x_new[j] -= step * direction
                x_new = np.clip(x_new, -0.49, 0.49)
                
                new_score = _min_radius(x_new)
                if new_score > current_r + 1e-12:
                    x = x_new
                    break
                step *= 0.5
        else:
            direction = -x[wall_idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-15:
                direction = np.array([0.1, 0.1, 0.1])
                norm = np.linalg.norm(direction)
            direction /= norm
            
            step = 0.01
            for _ in range(20):
                x_new = x.copy()
                x_new[wall_idx] += step * direction
                x_new = np.clip(x_new, -0.49, 0.49)
                
                new_score = _min_radius(x_new)
                if new_score > current_r + 1e-12:
                    x = x_new
                    break
                step *= 0.5
        
        current_score = _min_radius(x)
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
    
    return best_x, best_score


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    best_x = None
    best_score = -np.inf

    # Multiple initialization strategies
    inits = []
    
    # 1. Fibonacci lattice
    inits.append(_fibonacci_lattice_init(N, seed=42))
    
    # 2. Grid-based
    inits.append(_grid_init(N))
    
    # 3. Random
    inits.append(_random_init(N, seed=123))
    
    # 4. Best-first grid
    inits.append(_best_first_grid_init(N))
    
    # 5. Another random with different seed
    inits.append(_random_init(N, seed=777))
    
    # 6. Fibonacci with different seed
    inits.append(_fibonacci_lattice_init(N, seed=999))

    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 25:
            break

        x_init = x_init.copy()
        x_init = np.clip(x_init, -0.48, 0.48)

        # Phase 1: Quick repulsion-based optimization to get a reasonable starting point
        x_refined, score = _repulsion_optimize(x_init, max_iter=2000, time_limit=8.0)
        
        if score > best_score:
            best_score = score
            best_x = x_refined.copy()

        # Phase 2: Scipy refinement with beta annealing (the key improvement)
        if time.time() < deadline - 15:
            time_for_scipy = min(20.0, deadline - time.time() - 15.0)
            x_scipy, score_scipy = _scipy_refine_annealed(x_refined, time_limit=time_for_scipy)
            if score_scipy > best_score:
                best_score = score_scipy
                best_x = x_scipy.copy()

        # Phase 3: Polish
        if time.time() < deadline - 8:
            x_polished, score_polished = _polish(best_x, max_iter=1000, time_limit=6.0)
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished.copy()

    # Final polish on the best result
    if best_x is not None and time.time() < deadline - 4:
        x_final, score_final = _polish(best_x, max_iter=500, time_limit=4.0)
        if score_final > best_score:
            best_score = score_final
            best_x = x_final.copy()

    if best_x is None:
        best_x = _grid_init(N)

    # Final validity check
    best_x = np.clip(best_x, -0.5, 0.5)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-10:
                best_x[j] += 1e-5 * np.array([1, 0, 0])

    return best_x
# EVOLVE-BLOCK-END
