"""Packing 50 equal circles in a unit circle via ring-based parametric optimization with gradient polish."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _compute_score_fast(points):
    """Faster score computation using vectorized operations."""
    if len(points) != N or not np.all(np.isfinite(points)):
        return -1.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu])
    # Boundary
    norms = np.sqrt(np.sum(points**2, axis=1))
    min_margin = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_margin)

def _ring_initialization():
    """
    Generate initial ring-based configuration.
    Structure: 1 center + 6 inner ring + 12 second ring + 15 third ring + 16 outer ring = 50
    Parameters: [r1, r2, r3, r4, theta1, theta2, theta3, theta4]
    """
    # Ring counts: 1, 6, 12, 15, 16 = 50
    ring_counts = [1, 6, 12, 15, 16]
    
    # Initial radii for each ring (tuned for near-optimal packing)
    initial_r = [0.0, 0.25, 0.50, 0.72, 0.92]
    initial_theta = [0.0, 0.0, np.pi/6, np.pi/12, 0.0]
    
    params = initial_r + initial_theta
    points = _params_to_points(params)
    return params, points

def _params_to_points(params):
    """Convert ring parameters to 50 point coordinates."""
    ring_counts = [1, 6, 12, 15, 16]
    radii = params[:5]
    thetas = params[5:10]
    
    points = []
    for ring_idx, (count, r, theta) in enumerate(zip(ring_counts, radii, thetas)):
        if count == 1:
            points.append([r * np.cos(theta), r * np.sin(theta)])
        else:
            angles = theta + np.arange(count) * (2 * np.pi / count)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points, dtype=np.float64)

def _neg_score_ring(params):
    """Negative score for ring parameters (for minimization)."""
    points = _params_to_points(params)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _compute_score_fast(points)
    if s <= 0:
        return 1e10
    return -s

def _soft_min_surrogate(params, beta=30.0):
    """
    Smooth surrogate using log-sum-exp on all constraints.
    Returns negative soft-min radius.
    """
    points = _params_to_points(params)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - radii
    
    # All constraint distances / 2
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled = all_dists / 2.0
    
    # Soft min via log-sum-exp
    neg_scaled = -beta * scaled
    max_val = np.max(neg_scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_scaled - max_val)))
    soft_min = -log_sum_exp / beta
    
    return -soft_min

def _energy(params):
    """Energy function: sum of violations + soft penalty."""
    points = _params_to_points(params)
    if not np.all(np.isfinite(points)):
        return 1e10
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - radii
    
    # Target: maximize min(pair_dists/2, boundary_dists)
    # Energy = sum of max(0, violation) terms
    # We want all pair_dists >= 2*r and all boundary_dists >= r
    # Use a smooth approximation
    
    # For energy, use: -min + penalty for violations
    min_constraint = np.min(np.concatenate([pair_dists / 2.0, boundary_dists]))
    
    # Count violations
    pair_violations = np.sum(np.maximum(0, -pair_dists / 2.0))
    boundary_violations = np.sum(np.maximum(0, -boundary_dists))
    
    return -min_constraint + 10.0 * (pair_violations + boundary_violations)

def _energy_grad(params):
    """Gradient of energy function (numerical)."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = _energy(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        grad[i] = (_energy(params_plus) - f0) / eps
    return grad

def _global_search_ring_params(deadline, max_workers=None):
    """
    Use differential evolution to find good ring parameters.
    """
    # Bounds for each parameter
    # Radii: [0, 1.0] for ring 0 (center), [0.1, 0.95] for others
    # Angles: [0, 2*pi]
    bounds = [
        (0.0, 1.0),      # r0 (center)
        (0.1, 0.95),     # r1
        (0.3, 0.95),     # r2
        (0.5, 0.98),     # r3
        (0.7, 0.99),     # r4
        (0.0, 2*np.pi),  # theta0
        (0.0, 2*np.pi),  # theta1
        (0.0, 2*np.pi),  # theta2
        (0.0, 2*np.pi),  # theta3
        (0.0, 2*np.pi),  # theta4
    ]
    
    start_time = time.time()
    
    # Use differential evolution with time limit
    de_result = differential_evolution(
        _neg_score_ring,
        bounds,
        maxiter=200,
        popsize=15,
        tol=1e-8,
        seed=42,
        polish=True,
        init='sobol',
        workers=max_workers
    )
    
    elapsed = time.time() - start_time
    print(f"DE completed in {elapsed:.1f}s, best score: {-de_result.fun:.10f}")
    
    return de_result.x, -de_result.fun

def _local_search_ring_params(params, deadline):
    """
    Local search on ring parameters using gradient-based optimization.
    """
    start_time = time.time()
    
    # Try multiple beta values for soft-min
    best_params = params.copy()
    best_score = _compute_score_fast(_params_to_points(params))
    
    for beta in [20.0, 40.0, 80.0]:
        if time.time() > deadline:
            break
        
        result = minimize(
            _soft_min_surrogate,
            params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'beta': beta}
        )
        
        if time.time() > deadline:
            break
        
        new_params = result.x
        new_points = _params_to_points(new_params)
        
        if np.all(np.isfinite(new_points)):
            new_score = _compute_score_fast(new_points)
            if new_score > best_score:
                best_score = new_score
                best_params = new_params
    
    return best_params, best_score

def _polish_exact(points, deadline):
    """
    Final high-precision polish using exact score with L-BFGS-B.
    """
    start_time = time.time()
    
    # Use the soft-min surrogate with high beta for final polish
    # First, convert to flat parameter vector for the 100-dim space
    
    def objective_flat(x):
        pts = x.reshape(N, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        # High-beta soft min
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - radii
        all_dists = np.concatenate([pair_dists, boundary_dists])
        scaled = all_dists / 2.0
        beta = 100.0
        neg_scaled = -beta * scaled
        max_val = np.max(neg_scaled)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_scaled - max_val)))
        soft_min = -log_sum_exp / beta
        return -soft_min
    
    result = minimize(
        objective_flat,
        points.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
    )
    
    if time.time() > deadline:
        return points
    
    optimized = result.x.reshape(N, 2)
    if not np.all(np.isfinite(optimized)):
        return points
    
    # Validate
    radii = np.hypot(optimized[:, 0], optimized[:, 1])
    if np.any(radii > 1.0):
        return points
    
    new_score = _compute_score_fast(optimized)
    old_score = _compute_score_fast(points)
    
    if new_score > old_score:
        return optimized
    return points

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing as fallback."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Global search with differential evolution on ring parameters
    print("Phase 1: Differential evolution on ring parameters...")
    de_params, de_score = _global_search_ring_params(deadline)
    de_points = _params_to_points(de_params)
    de_actual_score = _compute_score_fast(de_points)
    
    if de_actual_score > best_score:
        best_score = de_actual_score
        best_points = de_points.copy()
        print(f"  DE best: {de_actual_score:.10f}")
    
    # Phase 2: Local search on ring parameters
    if time.time() < deadline - 20.0:
        print("Phase 2: Local search on ring parameters...")
        local_params, local_score = _local_search_ring_params(de_params, deadline)
        local_points = _params_to_points(local_params)
        local_actual = _compute_score_fast(local_points)
        
        if local_actual > best_score:
            best_score = local_actual
            best_points = local_points.copy()
            print(f"  Local best: {local_actual:.10f}")
    
    # Phase 3: Try multiple ring structures
    if time.time() < deadline - 15.0:
        print("Phase 3: Alternative ring structures...")
        
        # Alternative 1: 1+6+12+14+17
        alt_ring_counts = [1, 6, 12, 14, 17]
        alt_r = [0.0, 0.28, 0.52, 0.75, 0.93]
        alt_theta = [0.0, 0.0, np.pi/6, np.pi/14, 0.0]
        alt_params = alt_r + alt_theta
        
        # Optimize this structure
        result = minimize(
            _soft_min_surrogate,
            alt_params,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        if time.time() < deadline - 10.0:
            alt_points = _params_to_points(result.x)
            alt_score = _compute_score_fast(alt_points)
            if alt_score > best_score:
                best_score = alt_score
                best_points = alt_points.copy()
                print(f"  Alt1 best: {alt_score:.10f}")
        
        # Alternative 2: 1+7+12+14+16 (heavier inner ring)
        alt2_ring_counts = [1, 7, 12, 14, 16]
        alt2_r = [0.0, 0.26, 0.50, 0.73, 0.92]
        alt2_theta = [0.0, 0.0, np.pi/6, np.pi/14, 0.0]
        alt2_params = alt2_r + alt2_theta
        
        if time.time() < deadline - 5.0:
            result2 = minimize(
                _soft_min_surrogate,
                alt2_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if time.time() < deadline - 2.0:
                alt2_points = _params_to_points(result2.x)
                alt2_score = _compute_score_fast(alt2_points)
                if alt2_score > best_score:
                    best_score = alt2_score
                    best_points = alt2_points.copy()
                    print(f"  Alt2 best: {alt2_score:.10f}")
    
    # Phase 4: High-precision polish in full 100-dim space
    if best_points is not None and time.time() < deadline - 3.0:
        print("Phase 4: High-precision polish...")
        polished = _polish_exact(best_points, deadline)
        polished_score = _compute_score_fast(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished.copy()
            print(f"  Polished: {polished_score:.10f}")
    
    # Fallback if nothing worked
    if best_points is None:
        best_points = _hexagonal_initial()
    
    # Final validation
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    final_score = _compute_score_fast(best_points)
    print(f"Final score: {final_score:.10f}")
    
    return best_points
