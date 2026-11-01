"""Packing 200 equal circles in a unit square: Force-Directed + L-BFGS-B + Active Set Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 140.0  # safety margin before 150s kill

def _score(points):
    """Compute the packing radius for a set of points."""
    if len(points) < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    dist_bound = np.minimum(
        np.minimum(points[:, 0] + 0.5, 0.5 - points[:, 0]),
        np.minimum(points[:, 1] + 0.5, 0.5 - points[:, 1])
    )
    min_bound = np.min(dist_bound)
    return min(min_pair, min_bound)

def _hex_init(n, box_size=1.0, jitter=0.0):
    """Generate a hexagonal lattice of points within a square of side box_size."""
    # Hexagonal lattice: spacing s, row offset s*sqrt(3)/2
    s = np.sqrt(2.0 * box_size**2 / (n * np.sqrt(3)))
    num_cols = int(np.ceil(box_size / s)) + 2
    num_rows = int(np.ceil(box_size / (s * np.sqrt(3) / 2))) + 2
    
    points = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * s - (num_cols - 1) * s / 2.0
            y = row * s * np.sqrt(3) / 2.0 - (num_rows - 1) * s * np.sqrt(3) / 4.0
            if row % 2 == 1:
                x += s / 2
            points.append([x, y])
    points = np.array(points)
    
    # Filter inside box
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    elif len(points) < n:
        num_to_add = n - len(points)
        random_points = np.random.uniform(-0.5, 0.5, size=(num_to_add, 2))
        points = np.vstack([points, random_points])
    
    if jitter > 0:
        points += np.random.uniform(-jitter, jitter, size=points.shape)
    
    # Clip to box
    points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
    points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
    
    return points

def _random_init(n, box_size=1.0):
    """Generate random points within a square."""
    points = np.random.uniform(-0.5, 0.5, size=(n, 2))
    return points

def _force_directed(points, r, max_iters=300, step_size=0.3, time_limit=None, start_time=None):
    """
    Relax points to satisfy non-overlapping constraint for radius r.
    Uses force-directed method with repulsion for overlapping circles.
    """
    n = len(points)
    points = points.copy()
    step = step_size
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Check if constraints satisfied
        min_pair = np.min(dist)
        dist_bound = np.minimum(
            np.minimum(points[:, 0] + 0.5, 0.5 - points[:, 0]),
            np.minimum(points[:, 1] + 0.5, 0.5 - points[:, 1])
        )
        min_bound = np.min(dist_bound)
        
        if min_pair >= 2 * r and min_bound >= r:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion for overlapping pairs
        mask = dist < 2 * r
        mask[np.diag_indices_from(mask)] = False
        
        dist_safe = np.where(mask, dist, 1.0)
        unit_vecs = diff / dist_safe[..., np.newaxis]
        strength = np.where(mask, 2 * r - dist, 0.0)
        forces = np.sum(unit_vecs * strength[..., np.newaxis], axis=1)
        
        # Boundary forces
        dist_left = points[:, 0] + 0.5
        dist_right = 0.5 - points[:, 0]
        dist_bottom = points[:, 1] + 0.5
        dist_top = 0.5 - points[:, 1]
        
        forces[:, 0] += np.where(dist_left < r, r - dist_left, 0.0)
        forces[:, 0] -= np.where(dist_right < r, r - dist_right, 0.0)
        forces[:, 1] += np.where(dist_bottom < r, r - dist_bottom, 0.0)
        forces[:, 1] -= np.where(dist_top < r, r - dist_top, 0.0)
        
        # Update positions
        points += step * forces
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Decay step size
        step *= 0.99
        
    return points

def _soft_min_neg(x, beta=50.0):
    """
    Soft minimum of negative values using log-sum-exp trick.
    Returns a smooth approximation of min(x) for x <= 0.
    """
    # x is array of constraint violations (negative means satisfied)
    # We want to minimize the max violation, i.e., maximize min(x)
    # Use -soft_max(-x) = soft_min(x)
    x_max = np.max(x)
    if x_max > 0:
        # At least one constraint violated
        x_shifted = x - x_max
        return -x_max + np.log(np.sum(np.exp(beta * x_shifted))) / beta
    else:
        # All constraints satisfied
        return np.max(x)

def _soft_min_neg_grad(x, beta=50.0):
    """Gradient of soft_min_neg."""
    x_max = np.max(x)
    if x_max > 0:
        x_shifted = x - x_max
        weights = np.exp(beta * x_shifted)
        weights /= np.sum(weights)
        return weights
    else:
        # Subgradient: uniform over max elements
        max_mask = (x == x_max)
        n_max = np.sum(max_mask)
        grad = np.zeros_like(x)
        grad[max_mask] = 1.0 / n_max
        return grad

def _optimize_lbfgs(points, r, time_limit=None, start_time=None, max_iter=1000):
    """
    Optimize positions using L-BFGS-B to maximize minimum distance.
    Uses soft-min surrogate for the nonsmooth min objective.
    """
    n = len(points)
    
    def objective(flat_x):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                return 1e10  # large penalty
        pts = flat_x.reshape(n, 2)
        # Clip to box
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Constraint violations: want dist >= 2r, so violation = 2r - dist
        pair_viol = 2 * r - dist
        np.fill_diagonal(pair_viol, 0)
        pair_viol = pair_viol.flatten()
        
        # Boundary violations
        dist_left = pts[:, 0] + 0.5
        dist_right = 0.5 - pts[:, 0]
        dist_bottom = pts[:, 1] + 0.5
        dist_top = 0.5 - pts[:, 1]
        bound_viol = np.concatenate([
            r - dist_left, r - dist_right,
            r - dist_bottom, r - dist_top
        ])
        
        all_viol = np.concatenate([pair_viol, bound_viol])
        
        # We want to minimize the maximum violation
        # Use soft-max for smoothness
        return _soft_min_neg(-all_viol, beta=30.0)
    
    def gradient(flat_x):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                return np.zeros_like(flat_x)
        pts = flat_x.reshape(n, 2)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        pair_viol = 2 * r - dist
        np.fill_diagonal(pair_viol, 0)
        pair_viol_flat = pair_viol.flatten()
        
        dist_left = pts[:, 0] + 0.5
        dist_right = 0.5 - pts[:, 0]
        dist_bottom = pts[:, 1] + 0.5
        dist_top = 0.5 - pts[:, 1]
        bound_viol = np.concatenate([
            r - dist_left, r - dist_right,
            r - dist_bottom, r - dist_top
        ])
        
        all_viol = np.concatenate([pair_viol_flat, bound_viol])
        neg_viol = -all_viol
        
        # Gradient of soft_min_neg(-viol) = -gradient of soft_max(viol)
        # soft_max(viol) gradient = softmax(viol * beta)
        beta = 30.0
        viol_max = np.max(all_viol)
        if viol_max > 0:
            shifted = all_viol - viol_max
            weights = np.exp(beta * shifted)
            weights /= np.sum(weights)
        else:
            max_mask = (all_viol == viol_max)
            n_max = np.sum(max_mask)
            weights = np.zeros_like(all_viol)
            weights[max_mask] = 1.0 / n_max
        
        # d(soft_max(viol))/d(pts) = sum_j weights[j] * d(viol_j)/d(pts)
        # soft_min_neg(-viol) = -soft_max(viol), so gradient = -sum_j weights[j] * d(viol_j)/d(pts)
        
        grad_pts = np.zeros_like(pts)
        
        # Pairwise contribution
        n_pairs = n * n
        pair_weights = weights[:n_pairs].reshape(n, n)
        
        # d(2r - dist_ij)/d(pts_i) = (pts_i - pts_j) / dist_ij
        # d(2r - dist_ij)/d(pts_j) = (pts_j - pts_i) / dist_ij
        
        for i in range(n):
            for j in range(n):
                if i != j and pair_weights[i, j] > 0:
                    d = dist[i, j]
                    if d > 1e-10:
                        grad_pts[i] -= pair_weights[i, j] * (pts[i] - pts[j]) / d
                        grad_pts[j] -= pair_weights[i, j] * (pts[j] - pts[i]) / d
        
        # Boundary contribution
        # d(r - (x+0.5))/dx = -1, d(r - (0.5-x))/dx = 1
        # d(r - (y+0.5))/dy = -1, d(r - (0.5-y))/dy = 1
        n_bound = 4 * n
        bound_weights = weights[n_pairs:n_pairs + n_bound]
        
        for i in range(n):
            grad_pts[i, 0] += bound_weights[i] * 1.0        # r - (x+0.5) -> d/dx = -1, but we have -soft_max, so +1
            grad_pts[i, 0] -= bound_weights[n + i] * 1.0     # r - (0.5-x) -> d/dx = 1, so -1
            grad_pts[i, 1] += bound_weights[2*n + i] * 1.0   # r - (y+0.5) -> d/dy = -1, so +1
            grad_pts[i, 1] -= bound_weights[3*n + i] * 1.0   # r - (0.5-y) -> d/dy = 1, so -1
        
        return grad_pts.flatten()
    
    # Set bounds
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    result = minimize(
        objective,
        points.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
    )
    
    if result.success or result.fun < 1e10:
        return result.x.reshape(n, 2)
    return points

def _active_set_polish(points, r, max_iters=500, time_limit=None, start_time=None):
    """
    Polish using active set method: only consider constraints that are nearly active.
    """
    n = len(points)
    points = points.copy()
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        dist_bound = np.minimum(
            np.minimum(points[:, 0] + 0.5, 0.5 - points[:, 0]),
            np.minimum(points[:, 1] + 0.5, 0.5 - points[:, 1])
        )
        
        min_pair = np.min(dist)
        min_bound = np.min(dist_bound)
        
        if min_pair >= 2 * r - 1e-12 and min_bound >= r - 1e-12:
            break
        
        # Find active constraints (within tolerance)
        tol = 1e-6
        active_pair = dist < 2 * r + tol
        active_pair[np.diag_indices_from(active_pair)] = False
        
        active_bound_left = points[:, 0] + 0.5 < r + tol
        active_bound_right = 0.5 - points[:, 0] < r + tol
        active_bound_bottom = points[:, 1] + 0.5 < r + tol
        active_bound_top = 0.5 - points[:, 1] < r + tol
        
        # Compute forces from active constraints only
        forces = np.zeros_like(points)
        
        if np.any(active_pair):
            dist_safe = np.where(active_pair, dist, 1.0)
            unit_vecs = diff / dist_safe[..., np.newaxis]
            strength = np.where(active_pair, 2 * r - dist, 0.0)
            forces += np.sum(unit_vecs * strength[..., np.newaxis], axis=1)
        
        forces[:, 0] += np.where(active_bound_left, r - (points[:, 0] + 0.5), 0.0)
        forces[:, 0] -= np.where(active_bound_right, r - (0.5 - points[:, 0]), 0.0)
        forces[:, 1] += np.where(active_bound_bottom, r - (points[:, 1] + 0.5), 0.0)
        forces[:, 1] -= np.where(active_bound_top, r - (0.5 - points[:, 1]), 0.0)
        
        # Small step
        step = 0.1 * (1.0 + 0.5 * np.cos(it * 0.01))
        points += step * forces
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
    return points

def _high_precision_polish(points, r, max_iters=1000, time_limit=None, start_time=None):
    """
    High-precision polish using very small steps and tight tolerances.
    """
    n = len(points)
    points = points.copy()
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        dist_bound = np.minimum(
            np.minimum(points[:, 0] + 0.5, 0.5 - points[:, 0]),
            np.minimum(points[:, 1] + 0.5, 0.5 - points[:, 1])
        )
        
        min_pair = np.min(dist)
        min_bound = np.min(dist_bound)
        
        if min_pair >= 2 * r - 1e-14 and min_bound >= r - 1e-14:
            break
        
        # Find violated constraints
        tol = 1e-10
        active_pair = dist < 2 * r + tol
        active_pair[np.diag_indices_from(active_pair)] = False
        
        active_bound_left = points[:, 0] + 0.5 < r + tol
        active_bound_right = 0.5 - points[:, 0] < r + tol
        active_bound_bottom = points[:, 1] + 0.5 < r + tol
        active_bound_top = 0.5 - points[:, 1] < r + tol
        
        forces = np.zeros_like(points)
        
        if np.any(active_pair):
            dist_safe = np.where(active_pair, dist, 1.0)
            unit_vecs = diff / dist_safe[..., np.newaxis]
            strength = np.where(active_pair, 2 * r - dist, 0.0)
            forces += np.sum(unit_vecs * strength[..., np.newaxis], axis=1)
        
        forces[:, 0] += np.where(active_bound_left, r - (points[:, 0] + 0.5), 0.0)
        forces[:, 0] -= np.where(active_bound_right, r - (0.5 - points[:, 0]), 0.0)
        forces[:, 1] += np.where(active_bound_bottom, r - (points[:, 1] + 0.5), 0.0)
        forces[:, 1] -= np.where(active_bound_top, r - (0.5 - points[:, 1]), 0.0)
        
        # Very small adaptive step
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-15:
            step = min(0.05, 1e-4 / max_force)
            points += step * forces
            points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
            points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        else:
            break
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_points = None
    best_radius = 0.0
    
    # Multiple restarts with different initializations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            points = _hex_init(N, jitter=0.005)
        elif restart % 3 == 1:
            points = _hex_init(N, jitter=0.01)
        else:
            points = _random_init(N)
        
        # Initial force-directed relaxation
        current_r = _score(points)
        if current_r <= 0:
            current_r = 0.02
        
        # Phase 1: Force-directed to get close
        points = _force_directed(points, current_r, max_iters=200, step_size=0.3,
                                  time_limit=deadline - time.time() - 30, start_time=start_time)
        
        # Phase 2: L-BFGS-B optimization with soft-min surrogate
        remaining = deadline - time.time() - 10
        if remaining > 5:
            max_iter = min(500, int(remaining * 50))
            points = _optimize_lbfgs(points, current_r, time_limit=remaining, start_time=start_time,
                                      max_iter=max_iter)
        
        # Phase 3: Active set polish
        remaining = deadline - time.time() - 5
        if remaining > 2:
            points = _active_set_polish(points, current_r, max_iters=200,
                                         time_limit=remaining, start_time=start_time)
        
        # Phase 4: High-precision polish
        remaining = deadline - time.time() - 2
        if remaining > 1:
            points = _high_precision_polish(points, current_r, max_iters=300,
                                             time_limit=remaining, start_time=start_time)
        
        # Evaluate
        actual_r = _score(points)
        
        if actual_r > best_radius:
            best_radius = actual_r
            best_points = points.copy()
        
        # Try to increase radius slightly and re-optimize if time permits
        remaining = deadline - time.time() - 15
        if remaining > 10 and actual_r > 0.03:
            new_r = actual_r + 0.0002
            points_fd = _force_directed(points, new_r, max_iters=100, step_size=0.2,
                                         time_limit=remaining / 3, start_time=start_time)
            points_opt = _optimize_lbfgs(points_fd, new_r, time_limit=remaining / 3,
                                          start_time=start_time, max_iter=200)
            points_pol = _active_set_polish(points_opt, new_r, max_iters=100,
                                             time_limit=remaining / 3, start_time=start_time)
            
            new_actual_r = _score(points_pol)
            if new_actual_r > best_radius:
                best_radius = new_actual_r
                best_points = points_pol.copy()
    
    if best_points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_points
