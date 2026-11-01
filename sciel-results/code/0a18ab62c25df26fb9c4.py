import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)

def _get_clearances(centers):
    """Return array of all constraint clearances (pair distances/2 and boundary distances)."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    return np.concatenate([pair_dists, bound_dists]), i_idx, j_idx

def _soft_max_zero(x, eps=1e-4):
    """Smoothed max(0, x)."""
    return 0.5 * (x + np.sqrt(x*x + eps*eps))

def _soft_max_zero_grad(x, eps=1e-4):
    """Gradient of smoothed max(0, x)."""
    return 0.5 * (1 + x / np.sqrt(x*x + eps*eps))

def _objective_and_gradient(params, n, target_r, eps=1e-4):
    """
    Objective: Sum of squared soft violations of constraints.
    Violation for pair (i,j): max(0, target_r - d_ij/2)
    Violation for boundary i: max(0, target_r - (1 - ||c_i||))
    """
    centers = params.reshape(n, 2)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    
    # Clearances relative to target radius
    # We want d_ij/2 >= target_r  => d_ij >= 2*target_r
    # And 1 - ||c_i|| >= target_r => ||c_i|| <= 1 - target_r
    
    # Violations
    pair_violations = target_r * 2.0 - pair_dists  # Positive if violating
    bound_violations = target_r - bound_dists      # Positive if violating
    
    # Smoothed violations
    soft_pair = _soft_max_zero(pair_violations, eps)
    soft_bound = _soft_max_zero(bound_violations, eps)
    
    # Objective: Sum of squares
    obj = np.sum(soft_pair ** 2) + np.sum(soft_bound ** 2)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise gradient
    # d(obj)/d(centers) via chain rule
    # obj = sum (soft_viol)^2
    # d(obj)/d(soft_viol) = 2 * soft_viol
    # d(soft_viol)/d(viol) = soft_grad
    # d(viol)/d(centers)
    
    d_soft_pair = 2.0 * soft_pair * _soft_max_zero_grad(pair_violations, eps)
    
    # For each pair (i, j), viol = 2*r_target - d_ij
    # d(viol)/d(c_i) = - d(d_ij)/d(c_i) = -(c_i - c_j) / d_ij
    # d(viol)/d(c_j) = - d(d_ij)/d(c_j) = (c_i - c_j) / d_ij
    
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        d = pair_dists[k]
        if d < 1e-10:
            continue
        vec = (centers[i] - centers[j]) / d
        coeff = d_soft_pair[k]
        grad[i] -= coeff * vec
        grad[j] += coeff * vec
        
    # Boundary gradient
    d_soft_bound = 2.0 * soft_bound * _soft_max_zero_grad(bound_violations, eps)
    
    # For each i, viol = r_target - (1 - ||c_i||) = r_target - 1 + ||c_i||
    # d(viol)/d(c_i) = c_i / ||c_i||
    
    for i in range(n):
        norm = norms[i]
        if norm < 1e-10:
            continue
        vec = centers[i] / norm
        coeff = d_soft_bound[i]
        grad[i] += coeff * vec
        
    return obj, grad

def _get_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, seed=0):
    """Generate initial points on concentric rings."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 2: 12 points
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 3: 18 points
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    # Ring 4: remaining
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1)
        r = 0.9
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    return np.array(points[:n])

def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _optimize_single_start(centers, start_time, deadline, max_iters=5000):
    """
    Optimize from a single start using iterative target radius increase.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    target_r = best_score * 1.001
    eps = 1e-3
    
    for iteration in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Decrease eps to make constraints harder
        if iteration % 100 == 0 and eps > 1e-5:
            eps *= 0.8
            
        try:
            res = minimize(
                _objective_and_gradient,
                best_centers.flatten(),
                args=(N, target_r, eps),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-6}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if np.any(~np.isfinite(new_centers)):
                continue
                
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                # Increase target radius slightly
                target_r = best_score * 1.001
            else:
                # If we didn't improve, try to perturb or break
                # Check if we are stuck
                if res.fun < 1e-10:
                    # Constraints satisfied for current target_r
                    # Can we increase target_r?
                    target_r *= 1.005
                else:
                    # Stuck, try a small perturbation
                    rng = np.random.default_rng()
                    best_centers += rng.normal(0, 0.001, size=best_centers.shape)
                    # Ensure inside circle
                    norms = np.linalg.norm(best_centers, axis=1)
                    mask = norms > 1.0 - target_r
                    if np.any(mask):
                        best_centers[mask] *= (1.0 - target_r) / norms[mask]
                        
        except Exception:
            continue
            
    return best_centers, best_score

def _construct():
    """Main construction function."""
    start_time = time.time()
    
    # Generate multiple initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=1),
        _get_random_initialization(N, seed=999),
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Optimize from each initialization
    for init in initializations:
        if time.time() > DEADLINE - 20:
            break
            
        centers, score = _optimize_single_start(init, start_time, DEADLINE)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    if best_centers is None:
        return initializations[0]
        
    # Final polish: try to push a bit more with very small eps
    if time.time() < DEADLINE - 5:
        target_r = best_score * 1.0005
        try:
            res = minimize(
                _objective_and_gradient,
                best_centers.flatten(),
                args=(N, target_r, 1e-5),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            new_centers = res.x.reshape(N, 2)
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
        except Exception:
            pass

    return best_centers

def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
