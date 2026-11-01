"""Rewritten program for packing 50 equal circles in a unit circle.
Uses distance-field based optimization with constraint-focused local search."""
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


def _get_constraint_clearances(centers):
    """Return array of all constraint clearances (pair distances/2 and boundary distances)."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    return np.concatenate([pair_dists, bound_dists]), i_idx, j_idx


def _penalty_objective(params, n, penalty_weight, min_clearance):
    """
    Penalty-based objective: minimize sum of violated constraints.
    Encourages all constraints to be >= min_clearance.
    """
    centers = params.reshape(n, 2)
    clearances, i_idx, j_idx = _get_constraint_clearances(centers)
    
    # Violations: how much each constraint falls below min_clearance
    violations = np.maximum(0.0, min_clearance - clearances)
    
    # Penalty: sum of squared violations
    obj = penalty_weight * np.sum(violations ** 2)
    
    # Also add a small regularization to keep points well-spread
    # This helps avoid degenerate solutions
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    obj += 0.001 * np.sum(norms ** 2)
    
    return obj


def _penalty_gradient(params, n, penalty_weight, min_clearance):
    """Gradient of penalty objective."""
    centers = params.reshape(n, 2)
    clearances, i_idx, j_idx = _get_constraint_clearances(centers)
    
    violations = np.maximum(0.0, min_clearance - clearances)
    
    grad = np.zeros_like(centers)
    
    # Pair constraints
    pair_violations = violations[:n*(n-1)//2]
    for k in range(len(pair_violations)):
        v = pair_violations[k]
        if v > 0:
            i, j = i_idx[k], j_idx[k]
            d = np.linalg.norm(centers[i] - centers[j])
            if d > 1e-10:
                vec = (centers[i] - centers[j]) / d
                grad[i] += 2.0 * penalty_weight * v * vec / 2.0
                grad[j] -= 2.0 * penalty_weight * v * vec / 2.0
    
    # Boundary constraints
    bound_violations = violations[n*(n-1)//2:]
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    for i in range(n):
        v = bound_violations[i]
        if v > 0 and norms[i] > 1e-10:
            grad[i] -= 2.0 * penalty_weight * v * centers[i] / norms[i]
    
    # Regularization gradient
    grad += 0.002 * centers
    
    return obj, grad


def _separation_pass(centers, n_iter=100, step_size=0.01):
    """
    Iterative pairwise separation: for each pair that's too close,
    push them apart along the line connecting them.
    """
    centers = centers.copy()
    n = len(centers)
    target_min_dist = 2.0 * _compute_score(centers) * 0.95  # slightly below current score
    
    for _ in range(n_iter):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pairs
        min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_dist_idx
        if i == j:
            continue
        
        d = dists[i, j]
        if d < target_min_dist:
            # Push apart
            vec = (centers[i] - centers[j]) / d
            push = (target_min_dist - d) * 0.5 * step_size / target_min_dist
            centers[i] += vec * push
            centers[j] -= vec * push
        
        # Keep inside circle
        norms = np.linalg.norm(centers, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            centers[mask] *= 0.999 / norms[mask]
    
    return centers


def _adaptive_local_search(centers, start_time, deadline, n_iterations=2000):
    """
    Adaptive local search that focuses on the most constrained pairs.
    Uses larger steps for more violated constraints.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.default_rng(777)
    
    # Track which pairs are most constrained
    pair_constraint_count = np.zeros((N, N))
    
    for iteration in range(n_iterations):
        if time.time() > deadline - 5:
            break
        
        # Get current clearances
        clearances, i_idx, j_idx = _get_constraint_clearances(best_centers)
        min_clearance = np.min(clearances)
        
        # Find the most constrained pair
        min_idx = np.argmin(clearances)
        
        # Determine which constraint is limiting
        n_pairs = N * (N - 1) // 2
        if min_idx < n_pairs:
            # Pair constraint
            i, j = i_idx[min_idx], j_idx[min_idx]
            # Perturb both points
            direction = rng.choice([-1, 1])
            delta = rng.normal(0, 0.005, size=2)
            trial = best_centers.copy()
            trial[i] += direction * delta
            trial[j] -= direction * delta
        else:
            # Boundary constraint
            i = min_idx - n_pairs
            # Push inward
            norm = np.linalg.norm(best_centers[i])
            if norm > 1e-10:
                delta = -0.005 * best_centers[i] / norm * rng.uniform(0.5, 1.5)
                trial = best_centers.copy()
                trial[i] += delta
            else:
                continue
        
        # Ensure inside circle
        norms = np.linalg.norm(trial, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            trial[mask] *= 0.999 / norms[mask]
        
        trial_score = _compute_score(trial)
        
        if trial_score > best_score:
            best_score = trial_score
            best_centers = trial.copy()
            # Update constraint counts
            if min_idx < n_pairs:
                pair_constraint_count[i, j] += 1
                pair_constraint_count[j, i] += 1
            else:
                pair_constraint_count[i, i] += 1
    
    return best_centers, best_score


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
    """Generate initial points on concentric rings matching known optimal structure."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points at r=0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 2: 12 points at r=0.5
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 3: 18 points at r=0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 4: remaining points at r=0.9
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1) + np.pi / max(remaining, 1)
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


def _get_golden_angle_initialization(n, seed=0):
    """Generate points using golden angle spiral."""
    rng = np.random.default_rng(seed)
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        r = np.sqrt((i + 0.5) / n) * 0.95
        theta = i * golden_angle + rng.uniform(-0.05, 0.05)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    return np.array(points)


def _optimize_penalty(centers, start_time, deadline):
    """
    Optimize using penalty method with increasing penalty weights.
    Start with low penalty to find feasible region, then increase.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # First, do separation pass to get a better starting point
    best_centers = _separation_pass(best_centers, n_iter=50, step_size=0.02)
    best_score = _compute_score(best_centers)
    
    # Penalty schedule: start low, increase gradually
    penalty_schedule = [1.0, 5.0, 20.0, 50.0, 100.0, 200.0]
    
    for pw in penalty_schedule:
        if time.time() > deadline - 5:
            break
        
        # Target: current best score plus a small margin
        target = best_score * 1.01
        
        try:
            res = minimize(
                _penalty_objective,
                best_centers.flatten(),
                args=(N, pw, target),
                method='L-BFGS-B',
                options={'maxiter': 150, 'gtol': 1e-6}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if np.any(~np.isfinite(new_centers)):
                continue
            
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
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
        _get_golden_angle_initialization(N, seed=0),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=1),
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Optimize from each initialization using penalty method
    for init in initializations:
        if time.time() > DEADLINE - 30:
            break
        
        centers, score = _optimize_penalty(init, start_time, DEADLINE)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    if best_centers is None:
        return initializations[0]
    
    # Adaptive local search
    if time.time() < DEADLINE - 15:
        best_centers, best_score = _adaptive_local_search(best_centers, start_time, DEADLINE)
    
    # Final separation polish
    if time.time() < DEADLINE - 5:
        best_centers = _separation_pass(best_centers, n_iter=30, step_size=0.005)
        final_score = _compute_score(best_centers)
        if final_score > best_score:
            best_score = final_score
    
    return best_centers


def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
