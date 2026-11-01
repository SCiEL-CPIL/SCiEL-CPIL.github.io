"""Packing 300 circles in a unit circle using smooth soft-min optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n, seed=0):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    rng = np.random.RandomState(seed)
    
    # Estimate spacing for n points in unit circle
    # Area per point ~ pi/n, hex packing density ~ pi/sqrt(12)
    # spacing s: area per point ~ s^2 * sqrt(3)/2
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate hex lattice points
    max_k = int(1.0 / (s * 0.5)) + 3
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices].copy()
    else:
        # Fallback: random placement
        angles = rng.uniform(0, 2*np.pi, n)
        radii = rng.uniform(0, 0.95, n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 1e-5, points.shape)
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(points, axis=1)
    outside = norms > 0.9999
    if np.any(outside):
        points[outside] *= 0.9999 / norms[outside, np.newaxis]
    
    return points

def _soft_min_and_grad(x, beta):
    """
    Compute soft-min of pairwise distances/2 and boundary distances,
    along with its gradient.
    
    The soft-min is: -1/beta * log(sum(exp(-beta * d_i)))
    where d_i are the distances (pairwise/2 and boundary).
    """
    p = x.reshape(-1, 2)
    
    # Pairwise distances (upper triangle only)
    iu = np.triu_indices(N, k=1)
    diff = p[iu[0]] - p[iu[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1)) / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, bound_dists])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * d_i)))
    # For numerical stability, subtract max
    neg_beta_dists = -beta * all_dists
    max_val = np.max(neg_beta_dists)
    exp_vals = np.exp(neg_beta_dists - max_val)
    sum_exp = np.sum(exp_vals)
    soft_min_val = -(max_val + np.log(sum_exp)) / beta
    
    # Weights for gradient
    weights = exp_vals / sum_exp
    
    # Gradient
    grad = np.zeros_like(p)
    
    # Pairwise part
    w_pairs = weights[:len(pair_dists)]
    safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    coeff = w_pairs / safe_pair_dists
    vecs = p[iu[0]] - p[iu[1]]
    
    np.add.at(grad, iu[0], coeff[:, np.newaxis] * vecs)
    np.add.at(grad, iu[1], -coeff[:, np.newaxis] * vecs)
    
    # Boundary part
    w_bounds = weights[len(pair_dists):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    
    return soft_min_val, -grad  # Negate because we want to maximize the min distance

def _objective_and_grad(x, beta):
    """Objective and gradient for L-BFGS (minimize negative soft-min)."""
    val, grad = _soft_min_and_grad(x, beta)
    return -val, -grad  # Minimize negative = maximize

def _local_optimize(points, max_time=50.0):
    """
    Optimize using L-BFGS with multiple beta values (simulated annealing on beta).
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    current = points.copy().flatten()
    best = points.copy()
    best_score = _compute_score(points)
    
    # Beta values from low to high (annealing schedule)
    betas = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current,
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'gtol': 1e-10,
                    'ftol': 1e-12,
                    'maxls': 20,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best = candidate.copy()
                current = res.x.copy()
        except Exception:
            pass
    
    return best, best_score

def _final_polish(points, max_time=20.0):
    """
    Final high-precision polish with very high beta and tight tolerances.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    current = points.copy().flatten()
    best = points.copy()
    best_score = _compute_score(points)
    
    # Very high beta for near-exact min
    betas = [50000.0, 200000.0]
    
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current,
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'gtol': 1e-12,
                    'ftol': 1e-14,
                    'maxls': 30,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best = candidate.copy()
                current = res.x.copy()
        except Exception:
            pass
    
    return best, best_score

def _perturb_and_reoptimize(points, max_time=30.0, seed=0):
    """
    Perturb the best solution and re-optimize to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + max_time
    rng = np.random.RandomState(seed)
    
    current = points.copy()
    # Small random perturbation
    current += rng.normal(0, 0.01, current.shape)
    
    # Ensure inside unit circle
    norms = np.linalg.norm(current, axis=1)
    outside = norms > 0.9999
    if np.any(outside):
        current[outside] *= 0.9999 / norms[outside, np.newaxis]
    
    optimized, score = _local_optimize(current, max_time=max_time)
    return optimized, score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multiple initializations with local optimization
    init_seeds = [0, 1, 2, 3, 4]
    for seed in init_seeds:
        if time.time() > deadline - 60:
            break
        
        try:
            points = _hex_lattice_init(N, seed=seed)
            optimized, score = _local_optimize(points, max_time=25.0)
            
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
        except Exception:
            pass
    
    # Phase 2: Perturb and re-optimize from best
    if best_points is not None and time.time() < deadline - 40:
        for seed in range(3):
            if time.time() > deadline - 25:
                break
            
            try:
                perturbed, score = _perturb_and_reoptimize(
                    best_points, max_time=15.0, seed=seed + 100
                )
                if score > best_score:
                    best_score = score
                    best_points = perturbed.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            polished, polished_score = _final_polish(best_points, max_time=15.0)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
        except Exception:
            pass
    
    # Ensure points are inside the unit circle
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            best_points[outside] /= norms[outside, np.newaxis]
            best_score = _compute_score(best_points)
    
    # Fallback if something went wrong
    if best_points is None or best_score <= 0:
        best_points = _hex_lattice_init(N, seed=42)
        best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
