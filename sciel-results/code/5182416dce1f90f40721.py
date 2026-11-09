"""Fifty equal circles in a square – high-precision packing optimizer."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s kill


def _score_from_centres(c):
    """Compute the exact radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    
    # Pairwise distances
    # Using pdist for efficiency
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    
    # Distance to walls
    # dist to left: x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    
    r = min(min_pair / 2.0, min_wall)
    return r


def _min_distance_fast(c):
    """Fast computation of minimum distance using KD-tree."""
    tree = cKDTree(c)
    pairs = tree.query_pairs(2 * 0.1)  # generous radius for initial search
    if len(pairs) == 0:
        # Fallback: use pdist
        d = pdist(c)
        min_pair = d.min() if d.size > 0 else 0.0
    else:
        min_pair = min(np.linalg.norm(c[i] - c[j]) for i, j in pairs)
    
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    
    return min(min_pair / 2.0, min_wall)


def _epigraph_objective(x, beta=100.0):
    """
    Epigraph formulation: minimize -t + penalty for violations.
    x = [c_1, ..., c_N, t] where c_i are 2D centers and t is the radius.
    
    We want to maximize t such that:
    - All pairwise distances >= 2t
    - All wall distances >= t
    
    This is equivalent to minimizing -t subject to constraints.
    We use a smooth penalty for constraint violations.
    """
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Clip centers to stay in bounds
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    # Pairwise distances / 2
    d = pdist(c) / 2.0
    
    # Wall distances
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    wall_dists_flat = wall_dists.flatten()
    
    # All constraints: dist_i >= t
    all_dists = np.concatenate([d, wall_dists_flat])
    
    # Violations: max(0, t - dist_i)
    violations = np.maximum(0.0, t - all_dists)
    
    # Smooth penalty: sum of exp(-beta * violation) for violated constraints
    # This is 0 when no violations, large when violations exist
    if violations.size > 0:
        max_v = violations.max()
        if max_v > 1e-15:
            # Use soft penalty
            penalty = np.exp(-beta * violations).sum() / beta
        else:
            penalty = 0.0
    else:
        penalty = 0.0
    
    return -t + penalty


def _negative_smooth_score(c_flat, beta=100.0):
    """Objective to minimise (legacy, kept for compatibility)."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    d = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    wall_dists_flat = wall_dists.flatten()
    all_dists = np.concatenate([d, wall_dists_flat])
    
    min_d = all_dists.min()
    exp_d = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(exp_d.sum()) / beta
    
    return -soft_min


def _grid_init(rows, cols):
    """Generate a grid initial layout."""
    # Try to find best R, C such that R*C >= N
    best_r, best_c = 7, 8
    for r in range(5, 10):
        for c in range(5, 10):
            if r * c >= N:
                # Prefer square-ish
                if abs(r-c) < abs(best_r-best_c):
                    best_r, best_c = r, c
    
    x = np.linspace(-HALF + 0.05, HALF - 0.05, best_c)
    y = np.linspace(-HALF + 0.05, HALF - 0.05, best_r)
    
    # Create full grid
    full_grid = np.array([(xi, yi) for yi in y for xi in x])
    
    # Prioritize keeping center
    dists_to_center = np.sum(full_grid**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = full_grid[sorted_idx[:N]]
    return pts


def _hex_init():
    """Hexagonal packing initial layout."""
    points = []
    # Hexagonal close packing
    r_init = 0.071
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    max_x = HALF - r_init
    max_y = HALF - r_init
    
    row = 0
    y = -max_y
    while y <= max_y:
        x_start = -max_x
        if row % 2 == 1:
            x_start += dx / 2
        x = x_start
        while x <= max_x:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
        if len(points) > N * 3:
            break
            
    pts = np.array(points)
    dists_to_center = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = pts[sorted_idx[:N]]
    return pts


def _random_init(rng):
    return rng.uniform(-HALF, HALF, size=(N, 2))


def _optimize_with_epigraph(x0, beta_start=50.0, beta_end=500.0, n_stages=3):
    """
    Multi-stage optimization using epigraph formulation.
    Returns optimized centers.
    """
    # Initial t estimate
    c0 = x0.reshape(N, 2)
    r0 = _score_from_centres(c0)
    t0_init = max(r0, 0.01)
    
    x = np.concatenate([c0.ravel(), [t0_init]])
    
    # Bounds for centers
    bounds_c = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    # Bounds for t
    bounds_t = [(0.0, HALF)]
    bounds = bounds_c + bounds_t
    
    # Multi-stage with increasing beta
    betas = np.linspace(beta_start, beta_end, n_stages)
    
    for beta in betas:
        try:
            res = minimize(
                lambda x_: _epigraph_objective(x_, beta=beta),
                x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'iprint': 0}
            )
            x = res.x
        except Exception:
            pass
    
    # Extract centers
    c = x[:2*N].reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    return c


def _polish_powell(c, timeout=10.0):
    """
    Final high-precision polish using Powell's method to maximize the minimum distance.
    This handles the non-smooth min function and is often faster than Nelder-Mead in high dimensions.
    """
    t_start = time.time()
    
    def objective(x):
        c_local = x.reshape(N, 2)
        c_local = np.clip(c_local, -HALF + 1e-12, HALF - 1e-12)
        # We want to maximize r, so minimize -r
        d = pdist(c_local)
        min_pair = d.min() if d.size > 0 else 0.0
        
        dist_walls = np.min(np.stack([
            c_local[:, 0] + HALF,
            HALF - c_local[:, 0],
            c_local[:, 1] + HALF,
            HALF - c_local[:, 1]
        ]), axis=0)
        min_wall = dist_walls.min()
        
        r = min(min_pair / 2.0, min_wall)
        return -r

    x0 = c.ravel()
    
    # Powell options
    options = {
        'maxiter': 500,
        'xtol': 1e-12,
        'ftol': 1e-12,
        'disp': False
    }
    
    try:
        res = minimize(
            objective,
            x0,
            method='Powell',
            options=options
        )
        c_opt = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        
        # Verify it's better
        s_opt = _score_from_centres(c_opt)
        s_init = _score_from_centres(c)
        
        if s_opt > s_init:
            return c_opt
        else:
            return c
            
    except Exception:
        return c

def _polish_high_precision(c, max_iter=1000):
    """
    Final high-precision polish using very high beta soft-min.
    """
    x0 = c.ravel()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    # Use progressively higher beta
    for beta in [1000.0, 2000.0, 5000.0]:
        try:
            res = minimize(
                lambda cf: _negative_smooth_score(cf, beta=beta),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-16, 'iprint': 0}
            )
            x0 = res.x
        except Exception:
            pass
    
    c = np.clip(x0.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
    return c


def _simulated_annealing(c_init, t_start, t_end, n_steps, rng, time_budget=None):
    """
    Simulated annealing / Metropolis-Hastings to explore the energy landscape.
    Energy = -min_distance (we minimize this, i.e., maximize min distance).
    """
    c = c_init.copy()
    best_c = c.copy()
    best_score = _score_from_centres(best_c)
    
    # Build KD-tree for fast neighbor queries
    def get_energy(c_local):
        """Compute energy = -min_distance."""
        tree = cKDTree(c_local)
        # Query nearest neighbors
        dists, _ = tree.query(c_local, k=2)  # k=2 includes self
        min_pair = dists[:, 1].min()
        
        dist_walls = np.min(np.stack([
            c_local[:, 0] + HALF,
            HALF - c_local[:, 0],
            c_local[:, 1] + HALF,
            HALF - c_local[:, 1]
        ]), axis=0)
        min_wall = dist_walls.min()
        
        r = min(min_pair / 2.0, min_wall)
        return -r
    
    energy = get_energy(c)
    best_energy = energy
    
    for step in range(n_steps):
        # Check time budget
        if time_budget is not None and time.time() - time_budget[0] > time_budget[1]:
            break
        
        # Temperature schedule (geometric cooling)
        T = t_start * (t_end / t_start) ** (step / max(1, n_steps - 1))
        
        # Choose a random particle to move
        i = rng.integers(0, N)
        
        # Propose a move: small random displacement
        step_size = 0.02 * T / max(0.01, t_start) + 0.001
        delta = rng.normal(0, step_size, size=2)
        new_c = c.copy()
        new_c[i] = np.clip(c[i] + delta, -HALF + 1e-9, HALF - 1e-9)
        
        # Compute energy difference
        new_energy = get_energy(new_c)
        dE = new_energy - energy
        
        # Metropolis criterion
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-10)):
            c = new_c
            energy = new_energy
            
            if energy < best_energy:
                best_energy = energy
                best_c = c.copy()
                best_score = _score_from_centres(best_c)
    
    return best_c, best_score


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initial configurations
    inits = [
        _grid_init(7, 8),
        _hex_init(),
        _random_init(rng)
    ]
    
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    # Phase 1: Simulated annealing to explore landscape
    sa_time_budget = 60.0
    if time.time() - t0 < DEADLINE - sa_time_budget - 10:
        try:
            for init in inits:
                if time.time() - t0 > DEADLINE - 20:
                    break
                c_sa, score_sa = _simulated_annealing(
                    init, 
                    t_start=0.1, 
                    t_end=0.001, 
                    n_steps=5000,
                    rng=rng,
                    time_budget=(t0, DEADLINE - 20)
                )
                if score_sa > best_score:
                    best_score = score_sa
                    best_centres = c_sa
        except Exception:
            pass
    
    # Phase 2: Optimization loop with epigraph formulation
    n_starts = 20
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 25:
            break
            
        if i < len(inits):
            x0 = inits[i].ravel()
        else:
            # Perturb best or random
            if i % 3 == 0:
                # Small perturbation of best
                x0 = best_centres.ravel() + rng.normal(0, 0.01, size=2*N)
            elif i % 3 == 1:
                # Medium perturbation
                x0 = best_centres.ravel() + rng.normal(0, 0.03, size=2*N)
            else:
                # Random start
                x0 = _random_init(rng).ravel()
                
        x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)
        
        try:
            # Step 1: Epigraph optimization
            c_opt = _optimize_with_epigraph(x0, beta_start=50.0, beta_end=500.0, n_stages=3)
            
            # Step 2: High-precision polish if time permits
            if time.time() - t0 < DEADLINE - 15:
                c_opt = _polish_high_precision(c_opt, max_iter=500)
            
            s = _score_from_centres(c_opt)
            if s > best_score:
                best_score = s
                best_centres = c_opt
                
        except Exception:
            pass

    # Phase 3: Final high-precision polish on best result
    if time.time() - t0 < DEADLINE - 5:
        try:
            best_centres = _polish_high_precision(best_centres, max_iter=500)
            final_score = _score_from_centres(best_centres)
            if final_score > best_score:
                best_score = final_score
        except Exception:
            pass

    # Phase 4: Powell polish for non-smooth refinement
    if time.time() - t0 < DEADLINE - 2:
        try:
            best_centres = _polish_powell(best_centres)
            final_score = _score_from_centres(best_centres)
            if final_score > best_score:
                best_score = final_score
        except Exception:
            pass

    # Final validation
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres
