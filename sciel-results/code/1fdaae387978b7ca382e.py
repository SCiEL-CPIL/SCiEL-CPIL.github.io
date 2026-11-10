import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 115.0  # Safety margin from 120s

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    """
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    if np.any(bound_dists < -1e-9):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def make_hex_init(noise_scale=0.01):
    """Create a jittered hexagonal lattice initialization."""
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    hex_pos = points[indices].copy()
    max_abs = np.max(np.abs(hex_pos))
    if max_abs > 0:
        hex_pos = hex_pos * (0.49 / max_abs)
    hex_pos += np.random.normal(0, noise_scale, size=hex_pos.shape)
    hex_pos = np.clip(hex_pos, -0.4999, 0.4999)
    return hex_pos

def soft_min_objective(x, beta=0.005):
    """
    Smooth surrogate for the max-min radius problem.
    Minimizes -soft_min of all pairwise distances/2 and boundary distances.
    """
    positions = x.reshape(N, 2)
    
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    # Pairwise distances
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    pair_dists_half = dists[np.triu_indices(N, k=1)] / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists_half, bound_dists])
    
    # Soft minimum using log-sum-exp
    # soft_min = -beta * log(sum(exp(-d/beta)))
    max_d = np.max(all_dists)
    exp_terms = np.exp(-(all_dists - max_d) / beta)
    log_sum = np.log(np.sum(exp_terms)) + max_d / beta
    soft_min = max_d - beta * log_sum
    
    return -soft_min

def objective_min_dist(x):
    """
    Objective: minimize the negative of the minimum distance (pairwise and boundary).
    We want to maximize the minimum of:
      - pairwise distances
      - distance to boundary for each point
    So we minimize -min(min_pair_dist, min_bound_dist).
    
    Actually, let's define:
      d_pair = minimum pairwise distance
      d_bound = minimum distance to boundary (0.5 - max(abs(coord)))
      r = min(d_pair/2, d_bound)
    We want to maximize r, so minimize -r.
    """
    positions = x.reshape(N, 2)
    
    # Boundary distances
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    min_bound = np.min(bound_dists)
    
    # Pairwise distances using cdist (C-optimized)
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_bound, min_dist / 2.0)
    
    return -r

def construct():
    start_time = time.time()
    
    best_radius = 0.0
    best_positions = None
    
    # Phase 1: Differential Evolution for global exploration
    # Use a bounded domain [-0.49, 0.49] for each coordinate
    bounds = [(-0.49, 0.49)] * (2 * N)
    
    try:
        # Run differential evolution with a time-limited strategy
        # Use a reasonable number of generations and population size
        de_result = differential_evolution(
            soft_min_objective,
            bounds,
            args=(0.005,),  # beta for soft min
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            init='latinhypercube',
            polish=True,
            callback=lambda x, convergence: (convergence or time.time() - start_time > DEADLINE - 60)
        )
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 30:
            new_pos = de_result.x.reshape(N, 2)
            r, valid = compute_radius(new_pos)
            if valid and r > best_radius:
                best_radius = r
                best_positions = new_pos.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement using Nelder-Mead on the smooth objective
    if best_positions is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 20:
            x0 = best_positions.flatten()
            try:
                res = minimize(
                    soft_min_objective,
                    x0,
                    args=(0.002,),  # tighter beta for refinement
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'maxfev': 5000,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                
                new_pos = res.x.reshape(N, 2)
                r, valid = compute_radius(new_pos)
                if valid and r > best_radius:
                    best_radius = r
                    best_positions = new_pos.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish with L-BFGS-B
    if best_positions is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            x0 = best_positions.flatten()
            try:
                res = minimize(
                    soft_min_objective,
                    x0,
                    args=(0.001,),  # very tight beta
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 500, 'maxls': 100, 'ftol': 1e-15}
                )
                
                new_pos = res.x.reshape(N, 2)
                r, valid = compute_radius(new_pos)
                if valid and r > best_radius:
                    best_radius = r
                    best_positions = new_pos.copy()
            except Exception:
                pass
    
    # Fallback: return hex init if nothing else worked
    if best_positions is None:
        best_positions = make_hex_init(0.01)
    
    return best_positions
