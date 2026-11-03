"""Packing 50 equal circles in a unit circle via ring-based parametric optimization + smooth polish."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
SEED = 42

def _score(points):
    """Compute the objective: minimum of half-min-pair-distance and min-margin."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    norms = np.hypot(points[:, 0], points[:, 1])
    min_margin = np.min(1.0 - norms)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_margin)

def _ring_to_points(params):
    """
    Convert ring parameters to 50 points.
    params layout:
      - For each ring k: radius_k, angle_offset_k
      - Ring counts: ring0=1 (center), ring1=6, ring2=12, ring3=12, ring4=19
    Total params: 2 * 5 = 10
    """
    ring_counts = [1, 6, 12, 12, 19]
    n_rings = len(ring_counts)
    points = np.zeros((N, 2))
    idx = 0
    for k in range(n_rings):
        radius = params[2 * k]
        angle_offset = params[2 * k + 1]
        count = ring_counts[k]
        if count == 1:
            # Center circle
            points[idx] = [0.0, 0.0]
            idx += 1
        else:
            for i in range(count):
                angle = angle_offset + i * 2 * np.pi / count
                points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
                idx += 1
    return points

def _ring_objective(params):
    """Negative score for ring-based parameterization."""
    points = _ring_to_points(params)
    if not np.all(np.isfinite(points)):
        return 1e10
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        # Penalty for points outside
        penalty = np.sum(np.maximum(norms - 1.0, 0.0) ** 2)
        return 1e10 + penalty
    min_margin = np.min(1.0 - norms)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    score = min(min_pair / 2.0, min_margin)
    return -score

def _optimize_ring_params(params0, deadline):
    """Optimize ring parameters using L-BFGS-B."""
    try:
        result = minimize(
            _ring_objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.fun < _ring_objective(params0):
            return result.x
    except Exception:
        pass
    return params0

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

def _polish_smooth(points, deadline):
    """Polish using smooth objective with L-BFGS-B, multiple passes with increasing beta."""
    best = points.copy()
    best_score = _score(best)
    
    try:
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

def _anneal(initial_points, max_iters=30000, t_start=0.03, t_end=1e-7, deadline=None):
    """Simulated annealing to optimize circle packing."""
    points = initial_points.copy()
    current_score = _score(points)
    best_local_points = points.copy()
    best_local_score = current_score
    
    cooling = (t_end / t_start) ** (1.0 / max_iters)
    
    base_step = 0.05
    accept_count = 0
    total_count = 0
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        recent_accept_rate = accept_count / max(total_count, 1)
        if recent_accept_rate > 0.6:
            base_step *= 1.1
        elif recent_accept_rate < 0.3:
            base_step *= 0.9
        base_step = np.clip(base_step, 0.01, 0.15)
        
        idx = np.random.randint(N)
        step_size = base_step * (0.8 + 0.4 * np.random.rand())
        angle = np.random.uniform(0, 2 * np.pi)
        delta = np.array([np.cos(angle), np.sin(angle)]) * step_size
        
        new_points = points.copy()
        new_points[idx] += delta
        
        norm = np.linalg.norm(new_points[idx])
        if norm > 1.0:
            new_points[idx] *= (1.0 - 1e-9) / norm
        
        new_score = _score(new_points)
        
        accepted = False
        if new_score > current_score:
            points = new_points
            current_score = new_score
            accepted = True
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_points = points.copy()
        else:
            temp = t_start * cooling**it
            delta_score = new_score - current_score
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / max(temp, 1e-10)):
                points = new_points
                current_score = new_score
                accepted = True
        
        if accepted:
            accept_count += 1
        total_count += 1
        
        if total_count % 100 == 0:
            accept_count = 0
            total_count = 0
    
    return best_local_points, best_local_score

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

def _ring_initialization():
    """Generate ring-based initial configuration."""
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]
    r1 = 0.25
    for i in range(6):
        angle = i * 2 * np.pi / 6
        points[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    r2 = 0.5
    for i in range(12):
        angle = (i + 0.5) * 2 * np.pi / 12
        points[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    r3 = 0.72
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
    r4 = 0.88
    for i in range(19):
        angle = (i + 0.5) * 2 * np.pi / 19
        points[31 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(SEED)
    
    best_points = None
    best_score = -1.0
    
    # Strategy 1: Ring-based parametric optimization with differential evolution
    if time.time() < deadline - 40.0:
        ring_counts = [1, 6, 12, 12, 19]
        n_rings = len(ring_counts)
        # Bounds for ring parameters: radius in [0, 1], angle in [0, 2*pi]
        bounds = []
        for k in range(n_rings):
            if k == 0:
                bounds.append((0.0, 0.0))  # Center radius fixed at 0
                bounds.append((0.0, 2 * np.pi))
            else:
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 2 * np.pi))
        
        # Use differential evolution to explore ring configurations
        try:
            de_result = differential_evolution(
                _ring_objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                seed=SEED,
                polish=False,
                init='latinhypercube'
            )
            if time.time() < deadline - 10.0:
                # Polish the DE result
                best_params = _optimize_ring_params(de_result.x, deadline)
                de_points = _ring_to_points(best_params)
                de_score = _score(de_points)
                
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 2: Ring-based initialization + SA + smooth polish
    if time.time() < deadline - 30.0:
        initial = _ring_initialization()
        remaining = deadline - time.time()
        time_budget = min(remaining * 0.5, 40.0)
        max_iters = int(time_budget * 200)
        max_iters = max(max_iters, 5000)
        result_points, result_score = _anneal(initial, max_iters=max_iters, deadline=deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
        
        if time.time() < deadline - 15.0:
            polished = _polish_smooth(result_points, deadline)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Strategy 3: Hexagonal initialization + SA + smooth polish
    if time.time() < deadline - 30.0:
        initial = _hexagonal_initial()
        remaining = deadline - time.time()
        time_budget = min(remaining * 0.5, 40.0)
        max_iters = int(time_budget * 200)
        max_iters = max(max_iters, 5000)
        result_points, result_score = _anneal(initial, max_iters=max_iters, deadline=deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
        
        if time.time() < deadline - 15.0:
            polished = _polish_smooth(result_points, deadline)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Strategy 4: Random restarts
    for restart in range(3):
        if time.time() > deadline - 25.0:
            break
        
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0.05, 0.85, N)
        initial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        remaining = deadline - time.time()
        time_budget = min(remaining * 0.4, 30.0)
        max_iters = int(time_budget * 200)
        max_iters = max(max_iters, 3000)
        result_points, result_score = _anneal(initial, max_iters=max_iters, deadline=deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
        
        if time.time() < deadline - 15.0:
            polished = _polish_smooth(result_points, deadline)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 5.0:
        final_polished = _polish_smooth(best_points, deadline)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished
    
    # Ensure valid output
    if best_points is None:
        best_points = _hexagonal_initial()
    
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points
