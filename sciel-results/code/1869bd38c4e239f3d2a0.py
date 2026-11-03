"""Packing 50 equal circles in a unit circle via simulated annealing with smooth optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(points):
    """Compute the objective: minimum of half-min-pair-distance and min-margin."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    # Compute margins to container
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = np.min(1.0 - norms)
    return min(min_pair_dist / 2.0, min_margin)

def _pairwise_distances(points):
    """Compute all pairwise distances."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def _score_fast(points):
    """Fast approximate score: check all pairs + boundary (exact but vectorized)."""
    norms = np.hypot(points[:, 0], points[:, 1])
    min_margin = np.min(1.0 - norms)
    # Check all pairs - with N=50 this is 1225 pairs, fast enough
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_margin)

def _soft_min_objective(points_flat, beta=100.0):
    """Smooth approximation of the negative minimum radius using log-sum-exp."""
    points_2d = points_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
    boundary_dists = 1.0 - radii
    
    # All distances (pair and boundary)
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled_dists = all_dists / 2.0
    
    # Soft minimum via log-sum-exp (numerically stable)
    neg_beta_scaled = -beta * scaled_dists
    max_val = np.max(neg_beta_scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_scaled - max_val)))
    soft_min = -log_sum_exp / beta
    
    return -soft_min

def _soft_min_objective_grad(points_flat, beta=100.0):
    """Gradient of the smooth objective."""
    points_2d = points_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu].copy()
    
    # Boundary distances
    radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
    boundary_dists = 1.0 - radii
    
    # All distances (pair and boundary)
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled_dists = all_dists / 2.0
    
    # Weights for soft-min
    neg_beta_scaled = -beta * scaled_dists
    max_val = np.max(neg_beta_scaled)
    exp_vals = np.exp(neg_beta_scaled - max_val)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient with respect to points
    grad = np.zeros_like(points_2d)
    
    # Pair contributions
    n_pairs = len(pair_dists)
    for k in range(n_pairs):
        i, j = iu[0][k], iu[1][k]
        w = weights[k] * beta / 2.0
        d = diff[i, j] / (dists[i, j] + 1e-15)
        grad[i] -= w * d
        grad[j] += w * d
    
    # Boundary contributions
    for i in range(N):
        w = weights[n_pairs + i] * beta / 2.0
        r = radii[i] + 1e-15
        grad[i] += w * points_2d[i] / r
    
    return -grad.flatten()

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _anneal(initial_points, max_iters=30000, t_start=0.03, t_end=1e-7, deadline=None):
    """Simulated annealing to optimize circle packing with fast scoring."""
    points = initial_points.copy()
    current_score = _score(points)
    best_local_points = points.copy()
    best_local_score = current_score
    
    # Cooling schedule
    cooling = (t_end / t_start) ** (1.0 / max_iters)
    
    # Adaptive step size
    base_step = 0.05
    accept_count = 0
    total_count = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        # Adaptive step size based on recent acceptance rate
        recent_accept_rate = accept_count / max(total_count, 1)
        if recent_accept_rate > 0.6:
            base_step *= 1.1
        elif recent_accept_rate < 0.3:
            base_step *= 0.9
        base_step = np.clip(base_step, 0.01, 0.15)
        
        # Generate a random perturbation
        idx = np.random.randint(N)
        step_size = base_step * (0.8 + 0.4 * np.random.rand())
        angle = np.random.uniform(0, 2 * np.pi)
        delta = np.array([np.cos(angle), np.sin(angle)]) * step_size
        
        new_points = points.copy()
        new_points[idx] += delta
        
        # Keep inside container
        norm = np.linalg.norm(new_points[idx])
        if norm > 1.0:
            new_points[idx] *= (1.0 - 1e-9) / norm
        
        # Use fast score (now exact for N=50)
        new_score = _score_fast(new_points)
        
        accepted = False
        if new_score > current_score:
            points = new_points
            current_score = new_score
            accepted = True
            exact_current = _score(points)
            if exact_current > best_local_score:
                best_local_score = exact_current
                best_local_points = points.copy()
        else:
            # Accept with probability based on temperature
            temp = t_start * cooling**it
            delta_score = new_score - current_score
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / max(temp, 1e-10)):
                points = new_points
                current_score = new_score
                accepted = True
        
        if accepted:
            accept_count += 1
        total_count += 1
        
        # Reset counters periodically
        if total_count % 100 == 0:
            accept_count = 0
            total_count = 0
    
    # Final exact score
    final_score = _score(best_local_points)
    return best_local_points, final_score

def _polish_with_smooth(points, deadline):
    """Polish using smooth objective with L-BFGS-B, multiple passes with increasing beta."""
    best = points.copy()
    best_score = _score(best)
    
    try:
        # Multiple polish passes with increasing beta for tighter convergence
        betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
        for beta in betas:
            if time.time() > deadline - 3.0:
                break
            
            result = minimize(
                _soft_min_objective,
                best.flatten(),
                args=(beta,),
                jac=_soft_min_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            
            if time.time() > deadline - 2.0:
                break
            
            optimized_points = result.x.reshape(N, 2)
            
            # Verify validity
            if not np.all(np.isfinite(optimized_points)):
                continue
            
            radii = np.hypot(optimized_points[:, 0], optimized_points[:, 1])
            if np.any(radii > 1.0):
                continue
            
            diff = optimized_points[:, np.newaxis, :] - optimized_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            iu = np.triu_indices(N, k=1)
            if np.any(dists[iu] < 1e-10):
                continue
            
            new_score = _score(optimized_points)
            if new_score > best_score:
                best_score = new_score
                best = optimized_points.copy()
                
    except Exception:
        pass
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_points = None
    best_score = -1.0
    
    # Multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            initial = _hexagonal_initial()
        elif restart == 1:
            # Ring-based initialization
            initial = _ring_initialization()
        elif restart == 2:
            # Modified ring with slight perturbation
            initial = _ring_initialization()
            initial += np.random.normal(0, 0.01, initial.shape)
        else:
            # Random initial configuration with some structure
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.random.uniform(0.05, 0.85, N)
            initial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Run annealing
        remaining = deadline - time.time()
        # Give more time to early restarts, less to later ones
        time_budget = min(remaining * 0.7, 60.0)
        max_iters = int(time_budget * 300)  # Scale iterations to remaining time
        max_iters = max(max_iters, 5000)
        result_points, result_score = _anneal(initial, max_iters=max_iters, deadline=deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
        
        # Polish with smooth optimization if time permits
        if time.time() < deadline - 8.0:
            polished = _polish_with_smooth(result_points, deadline)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Final high-precision polish if time remains
    if best_points is not None and time.time() < deadline - 2.0:
        final_polished = _polish_with_smooth(best_points, deadline)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished
    
    # Ensure valid output
    if best_points is None:
        best_points = _hexagonal_initial()
    
    # Final validation and clipping
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points

def _ring_initialization():
    """Generate ring-based initial configuration."""
    points = np.zeros((N, 2))
    # Center circle
    points[0] = [0.0, 0.0]
    # First ring: 6 circles
    r1 = 0.25
    for i in range(6):
        angle = i * 2 * np.pi / 6
        points[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    # Second ring: 12 circles
    r2 = 0.5
    for i in range(12):
        angle = (i + 0.5) * 2 * np.pi / 12
        points[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    # Third ring: 12 circles
    r3 = 0.72
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
    # Fourth ring: 19 circles
    r4 = 0.88
    for i in range(19):
        angle = (i + 0.5) * 2 * np.pi / 19
        points[31 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
    return points
