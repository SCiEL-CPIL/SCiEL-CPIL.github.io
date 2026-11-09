"""Fifty equal circles in a square – hybrid DE + L-BFGS-B optimizer."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

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
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    
    r = min(min_pair / 2.0, min_wall)
    return r


def _soft_radius(c, beta):
    """
    Compute a smooth approximation of the minimum radius.
    Uses the log-sum-exp trick for numerical stability.
    R_soft = min(d) - (1/beta) * log(sum(exp(-beta * (d - min(d)))))
    """
    c = c.reshape(N, 2)
    
    # Pairwise distances / 2
    d_pair = pdist(c) / 2.0
    
    # Wall distances
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1).flatten()
    
    all_dists = np.concatenate([d_pair, wall_dists])
    
    # If any distance is 0, radius is 0
    if np.any(all_dists < 1e-12):
        return 0.0
        
    min_d = np.min(all_dists)
    
    # log-sum-exp of -beta * (all_dists - min_d)
    # x = -beta * (all_dists - min_d)
    # max_x = 0 (since min_d is the minimum, all_dists - min_d >= 0, so x <= 0)
    # sum(exp(x))
    
    x = -beta * (all_dists - min_d)
    # x is <= 0. max(x) is 0.
    # log(sum(exp(x))) = log(sum(exp(x - 0)))
    
    # To avoid underflow in exp for large negative x, we can just compute it.
    # exp(-beta * large) -> 0.
    # sum(exp(x)) will be at least 1 (from the min element).
    
    s = np.sum(np.exp(x))
    
    return min_d - np.log(s) / beta


def _objective_de(x, beta):
    """Objective for Differential Evolution: minimize negative soft radius."""
    c = x.reshape(N, 2)
    # Clip to bounds to avoid issues, though DE handles bounds
    c = np.clip(c, -HALF, HALF)
    r = _soft_radius(c, beta)
    return -r


def _objective_polish(x, beta):
    """Objective for L-BFGS-B polish: minimize negative soft radius."""
    c = x.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-10, HALF - 1e-10)
    r = _soft_radius(c, beta)
    return -r


def _grid_init():
    """Generate a jittered grid initial layout."""
    # 7x8 grid
    rows, cols = 7, 8
    x = np.linspace(-HALF, HALF, cols)
    y = np.linspace(-HALF, HALF, rows)
    grid = np.array([(xi, yi) for yi in y for xi in x])
    
    # We have 56 points. Need 50.
    # Remove 6 points. To maintain symmetry, remove from corners/edges?
    # Or just take the first 50?
    # Better: Create a 5x10 grid? 50 points exactly.
    # 5 rows, 10 cols.
    x = np.linspace(-HALF, HALF, 10)
    y = np.linspace(-HALF, HALF, 5)
    grid = np.array([(xi, yi) for yi in y for xi in x])
    return grid


def _hex_init():
    """Hexagonal packing initial layout."""
    points = []
    r_init = 0.07
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
        if len(points) > N * 2:
            break
            
    pts = np.array(points)
    dists_to_center = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = pts[sorted_idx[:N]]
    return pts


def _random_init(rng):
    return rng.uniform(-HALF, HALF, size=(N, 2))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initial configurations
    inits = [
        _grid_init(),
        _hex_init(),
        _random_init(rng)
    ]
    
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    # Phase 1: Global Search with Differential Evolution
    # Use a moderate beta for smooth landscape
    beta_de = 20.0
    bounds = [(-HALF, HALF)] * (2 * N)
    
    # Run DE on a few initial points? DE is global, so it doesn't need good starts.
    # But we can run it once.
    # Time limit for DE: 60s
    de_deadline = t0 + 60.0
    
    try:
        # popsize=10 means 1000 individuals.
        # maxiter=30 means 30000 evaluations.
        # Each eval ~ 1ms? 30s.
        res_de = differential_evolution(
            lambda x: _objective_de(x, beta_de),
            bounds,
            seed=42,
            maxiter=30,
            popsize=10,
            tol=1e-6,
            polish=False,  # We do our own polish
            init='latinhypercube',
            # Update callback to check time?
            # DE doesn't have a direct time callback, but we can wrap it.
        )
        
        c_de = res_de.x.reshape(N, 2)
        c_de = np.clip(c_de, -HALF, HALF)
        score_de = _score_from_centres(c_de)
        
        if score_de > best_score:
            best_score = score_de
            best_centres = c_de
            
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Phase 2: Local Polish
    # Take the best candidate and polish with high beta L-BFGS-B
    # Also try polishing the initial grid/hex in case DE got stuck
    candidates = [best_centres, _grid_init(), _hex_init()]
    
    # Add some random perturbations of the best
    for _ in range(3):
        perturbed = best_centres + rng.normal(0, 0.01, size=(N, 2))
        perturbed = np.clip(perturbed, -HALF, HALF)
        candidates.append(perturbed)
    
    for c_init in candidates:
        if time.time() > DEADLINE - 10:
            break
            
        x0 = c_init.ravel()
        
        # Multi-stage polish with increasing beta
        for beta in [100.0, 500.0, 1000.0]:
            if time.time() > DEADLINE - 5:
                break
                
            try:
                res = minimize(
                    lambda x: _objective_polish(x, beta),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'iprint': 0}
                )
                x0 = res.x
            except Exception:
                pass
        
        c_opt = np.clip(x0.reshape(N, 2), -HALF, HALF)
        score_opt = _score_from_centres(c_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_centres = c_opt

    # Final validation
    best_centres = np.clip(best_centres, -HALF + 1e-12, HALF - 1e-12)
    return best_centres
