"""Rewritten program: multi-stage soft-min (log-sum-exp) optimization for 50-circle packing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _score(points):
    """Compute the packing radius (the objective to maximize)."""
    if points.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    # Check inside container
    dists_center = np.linalg.norm(points, axis=1)
    if np.any(dists_center > 1.0):
        return -1.0
    # Pairwise distances
    from scipy.spatial.distance import pdist
    pd = pdist(points)
    min_pair = np.min(pd) / 2.0
    min_boundary = 1.0 - np.max(dists_center)
    r = min(min_pair, min_boundary)
    if r <= 0:
        return -1.0
    return r


def _energy(points, beta):
    """
    Soft-min energy: -log-sum-exp of pairwise distances and boundary distances.
    We want to maximize the minimum distance, so we minimize the negative soft-min.
    """
    from scipy.spatial.distance import pdist
    pd = pdist(points)
    # Boundary distances
    dists_center = np.linalg.norm(points, axis=1)
    boundary = 1.0 - dists_center
    
    # All "distances" to consider (half pairwise + boundary)
    # For pairwise, the constraint is |ci-cj| >= 2r, so the relevant value is |ci-cj|/2
    half_pair = pd / 2.0
    # Boundary: 1 - |ci|
    
    all_vals = np.concatenate([half_pair, boundary])
    
    # Soft min: -log-sum-exp(-beta * all_vals) / beta
    # This approximates min(all_vals) as beta -> inf
    max_val = np.max(-beta * all_vals)
    lse = max_val + np.log(np.sum(np.exp(-beta * all_vals - max_val)))
    return lse / beta


def _energy_grad(points, beta):
    """Gradient of the soft-min energy w.r.t. points."""
    from scipy.spatial.distance import pdist
    n = points.shape[0]
    pd = pdist(points)
    dists_center = np.linalg.norm(points, axis=1)
    boundary = 1.0 - dists_center
    
    half_pair = pd / 2.0
    all_vals = np.concatenate([half_pair, boundary])
    
    neg_beta_all = -beta * all_vals
    max_val = np.max(neg_beta_all)
    exp_vals = np.exp(neg_beta_all - max_val)
    sum_exp = np.sum(exp_vals)
    # weights for soft-min
    weights = exp_vals / sum_exp
    
    grad = np.zeros_like(points)
    
    # Pairwise part
    n_pairs = len(half_pair)
    pair_weights = weights[:n_pairs]
    
    # Build pairwise index mapping
    # pdist gives distances in order: (0,1),(0,2),...,(0,n-1),(1,2),...
    for idx in range(n_pairs):
        i, j = np.unravel_index(idx, (n, n))
        if i >= j:
            continue
        w = pair_weights[idx]
        if w < 1e-15:
            continue
        diff = points[i] - points[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-15:
            continue
        # d(half_pair)/d(points[i]) = diff / (2 * dist)
        # d(energy)/d(points[i]) = -weight * d(half_pair)/d(points[i])
        # Actually energy = sum w_k * val_k (approximately, from log-sum-exp derivative)
        # d(energy)/d(points[i]) = -w * diff/(2*dist) for pair (i,j)
        grad[i] -= w * diff / (2.0 * dist)
        grad[j] += w * diff / (2.0 * dist)
    
    # Boundary part
    boundary_weights = weights[n_pairs:]
    for i in range(n):
        w = boundary_weights[i]
        if w < 1e-15:
            continue
        norm = dists_center[i]
        if norm < 1e-15:
            continue
        # boundary_i = 1 - |ci|
        # d(boundary_i)/d(ci) = -ci / |ci|
        # d(energy)/d(ci) = -w * d(boundary_i)/d(ci) = w * ci / |ci|
        grad[i] += w * points[i] / norm
    
    return grad


def _initial_hexagonal():
    """Generate initial hexagonal lattice configuration."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def _initial_ring():
    """Generate initial ring-based configuration."""
    # 1 center + rings
    points = [np.array([0.0, 0.0])]
    # Ring 1: 6 circles
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        points.append(np.array([0.35 * np.cos(angle), 0.35 * np.sin(angle)]))
    # Ring 2: 12 circles
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        points.append(np.array([0.7 * np.cos(angle), 0.7 * np.sin(angle)]))
    # Ring 3: 31 more to reach 50
    remaining = N - len(points)
    for i in range(remaining):
        angle = i * 2 * np.pi / remaining
        r = 0.85 + 0.05 * (i % 2)
        points.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    return np.array(points[:N])


def _initial_random():
    """Generate initial random configuration."""
    np.random.seed(42)
    points = np.random.randn(N, 2) * 0.8
    # Ensure they're not too close initially
    for _ in range(1000):
        from scipy.spatial.distance import pdist
        pd = pdist(points)
        if np.min(pd) > 0.05 and np.max(np.linalg.norm(points, axis=1)) < 0.95:
            break
        points = np.random.randn(N, 2) * 0.8
    return points


def construct():
    """
    Multi-stage soft-min optimization for circle packing.
    """
    start_time = time.time()
    
    # Try multiple initializations
    initializations = [
        _initial_hexagonal(),
        _initial_ring(),
        _initial_random(),
    ]
    
    best_points = None
    best_score = -1.0
    
    for init_idx, init_points in enumerate(initializations):
        if time.time() > _DEADLINE:
            break
        
        points = init_points.copy()
        
        # Multi-stage beta schedule
        # Start with low beta (smooth landscape) and increase
        beta_schedule = [10, 50, 100, 200, 500, 1000, 2000, 5000]
        
        current_points = points.copy()
        
        for beta in beta_schedule:
            if time.time() > _DEADLINE:
                break
            
            # Optimize with current beta
            result = minimize(
                _energy,
                current_points,
                jac=_energy_grad,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            current_points = result.x
            
            # Check score
            s = _score(current_points)
            if s > best_score:
                best_score = s
                best_points = current_points.copy()
        
        # Local refinement: try to improve the minimum distance specifically
        # Use the current best as starting point for a final stage with very high beta
        if time.time() < _DEADLINE - 5:
            final_result = minimize(
                _energy,
                current_points,
                jac=_energy_grad,
                args=(10000,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            current_points = final_result.x
            s = _score(current_points)
            if s > best_score:
                best_score = s
                best_points = current_points.copy()
    
    if best_points is None:
        best_points = _initial_hexagonal()
    
    # Final validation
    s = _score(best_points)
    if s <= 0:
        best_points = _initial_hexagonal()
    
    return best_points
# EVOLVE-BLOCK-END
