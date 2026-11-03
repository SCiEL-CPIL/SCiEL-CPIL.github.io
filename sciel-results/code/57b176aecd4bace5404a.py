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
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = np.min(1.0 - norms)
    return min(min_pair_dist / 2.0, min_margin)

def _soft_min_objective_and_grad(points_flat, beta=100.0):
    """Smooth approximation of negative min radius with gradient via log-sum-exp."""
    pts = points_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - radii
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled = all_dists / 2.0
    
    # Soft min via log-sum-exp: f = -log(sum(exp(-beta*scaled)))/beta
    neg_beta_scaled = -beta * scaled
    max_val = np.max(neg_beta_scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_scaled - max_val)))
    soft_min = -log_sum_exp / beta
    objective = -soft_min
    
    # Gradient computation
    # d(objective)/d(scaled_k) = (beta * exp(-beta*scaled_k)) / sum(exp(-beta*scaled_j))
    weights = np.exp(neg_beta_scaled - max_val)
    weights /= np.sum(weights)
    
    grad_scaled = -beta * weights  # d(objective)/d(scaled_k)
    
    # Backpropagate to points
    grad_pts = np.zeros_like(pts)
    
    # Pair contributions
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = grad_scaled[idx] / 2.0  # d(obj)/d(dist_ij)
        if dists[i, j] > 1e-12:
            d_dij = (pts[i] - pts[j]) / dists[i, j]
            grad_pts[i] += w * d_dij
            grad_pts[j] -= w * d_dij
    
    # Boundary contributions
    for i in range(N):
        w = grad_scaled[len(pair_dists) + i] / 2.0  # d(obj)/d(boundary_dist_i)
        if radii[i] > 1e-12:
            d_dr = pts[i] / radii[i]
            grad_pts[i] += w * d_dr
    
    return objective, grad_pts.flatten()

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

def _ring_initial():
    """Generate concentric ring initial configuration (1+6+12+18+13=50)."""
    rings = [(1, 0.0), (6, 0.15), (12, 0.30), (18, 0.45), (13, 0.60)]
    points = []
    for count, r in rings:
        if count == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.random.uniform(0, 2*np.pi/count)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])

def _anneal(initial_points, max_iters=30000, t_start=0.04, t_end=1e-9, deadline=None):
    """Simulated annealing to optimize circle packing."""
    points = initial_points.copy()
    current_score = _score(points)
    best_local_points = points.copy()
    best_local_score = current_score
    cooling = (t_end / t_start) ** (1.0 / max_iters)
    
    # Precompute indices for speed
    iu = np.triu_indices(N, k=1)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = np.random.randint(N)
        step_size = 0.015 * (1.0 + 0.5 * np.random.rand())
        angle = np.random.uniform(0, 2 * np.pi)
        delta = np.array([np.cos(angle), np.sin(angle)]) * step_size
        
        new_points = points.copy()
        new_points[idx] += delta
        
        norm = np.linalg.norm(new_points[idx])
        if norm > 1.0:
            new_points[idx] *= (1.0 - 1e-9) / norm
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_points = points.copy()
        else:
            temp = t_start * cooling**it
            delta_score = new_score - current_score
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / max(temp, 1e-10)):
                points = new_points
                current_score = new_score
        
        if it % 200 == 0 and deadline is not None and time.time() > deadline:
            break
    
    return best_local_points, best_local_score

def _polish_with_smooth(points, deadline, beta=100.0):
    """Polish using smooth objective with L-BFGS-B and gradient."""
    try:
        result = minimize(
            lambda x: _soft_min_objective_and_grad(x, beta)[0],
            lambda x: _soft_min_objective_and_grad(x, beta)[1],
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if time.time() > deadline:
            return points
        
        optimized_points = result.x.reshape(N, 2)
        
        if not np.all(np.isfinite(optimized_points)):
            return points
        
        radii = np.hypot(optimized_points[:, 0], optimized_points[:, 1])
        if np.any(radii > 1.0 + 1e-10):
            return points
        
        new_score = _score(optimized_points)
        old_score = _score(points)
        
        if new_score > old_score:
            return optimized_points
        else:
            return points
            
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    np.random.seed(42)
    
    best_points = None
    best_score = -1.0
    
    # Try multiple initializations
    initializers = [
        _hexagonal_initial,
        _ring_initial,
        lambda: np.column_stack([
            np.random.uniform(0.1, 0.85, N) * np.cos(np.random.uniform(0, 2*np.pi, N)),
            np.random.uniform(0.1, 0.85, N) * np.sin(np.random.uniform(0, 2*np.pi, N))
        ])
    ]
    
    for init_func in initializers:
        if time.time() > deadline:
            break
        
        initial = init_func()
        result_points, result_score = _anneal(initial, max_iters=25000, deadline=deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
        
        if time.time() < deadline - 5.0:
            polished = _polish_with_smooth(result_points, deadline)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 2.0:
        final_polished = _polish_with_smooth(best_points, deadline, beta=200.0)
        final_score = _score(final_polished)
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished
    
    if best_points is None:
        best_points = _hexagonal_initial()
    
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points
