"""Packing 50 equal circles in a unit circle: structural rewrite using multi-start
force-directed optimization with soft-min objective and annealing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # seconds; leave margin before 120s limit


def _score(pts):
    """Compute the packing radius for N points in the unit disk."""
    n = len(pts)
    if n != N:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Boundary constraint
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    min_boundary = 1.0 - np.max(radii)
    return min(min_pair, min_boundary)


def _objective(pts, beta):
    """Soft-min objective: -log(sum(exp(-beta * d_ij))) for pairs, plus boundary.
    Lower is better. We maximize the min distance, so we minimize -soft_min."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Soft min over pair distances
    # Use log-sum-exp for stability
    min_d = np.min(dists)
    log_sum = min_d * beta + np.log(np.sum(np.exp(-beta * (dists - min_d))))
    soft_min_pair = (min_d - np.log(np.sum(np.exp(-beta * (dists - min_d))))) / beta
    # Actually let me redo this more carefully
    # soft_min = -(1/beta) * log(sum(exp(-beta * d)))
    # = -(1/beta) * log(sum(exp(-beta * (d - min_d)) * exp(-beta * min_d)))
    # = -(1/beta) * [log(sum(exp(-beta * (d - min_d)))) + (-beta * min_d)]
    # = min_d - (1/beta) * log(sum(exp(-beta * (d - min_d))))
    
    # Boundary terms
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    boundary_dists = 1.0 - radii
    min_b = np.min(boundary_dists)
    log_sum_b = min_b * beta + np.log(np.sum(np.exp(-beta * (boundary_dists - min_b))))
    soft_min_boundary = min_b - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (boundary_dists - min_b))))
    
    # Combined: we want to maximize min(soft_min_pair, soft_min_boundary)
    # Use soft min of the two
    combined = min(soft_min_pair, soft_min_boundary)
    return -combined


def _grad_objective(pts, beta):
    """Gradient of the objective."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Soft min pair gradient
    exp_vals = np.exp(-beta * (dists - np.min(dists)))
    exp_vals[np.diag_indices(n)] = 0
    weights = exp_vals / exp_vals.sum()
    
    # d(soft_min)/d(pts_i) = -sum_j weights_ij * (pts_i - pts_j) / dists_ij
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = dists[i, j]
                if d > 1e-10:
                    grad[i] -= weights[i, j] * (pts[i] - pts[j]) / d
    
    # Boundary gradient
    radii = np.sqrt(np.sum(pts * pts, axis=1))
    boundary_dists = 1.0 - radii
    exp_b = np.exp(-beta * (boundary_dists - np.min(boundary_dists)))
    weights_b = exp_b / exp_b.sum()
    for i in range(n):
        r = radii[i]
        if r > 1e-10:
            grad[i] -= weights_b[i] * pts[i] / r
    
    return grad


def _hex_lattice_init(n_points):
    """Generate a hexagonal lattice initial configuration."""
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.sqrt(x * x + y * y) <= 1.0 - spacing / 2.0:
                rows.append([x, y])
    rows = np.array(rows)
    if len(rows) < n_points:
        return _random_init(n_points)
    # Pick the n_points closest to center
    radii = np.sqrt(np.sum(rows * rows, axis=1))
    idx = np.argsort(radii, kind='stable')[:n_points]
    return rows[idx]


def _ring_init(n_points):
    """Initialize points on concentric rings."""
    # 1 center + rings
    pts = [[0.0, 0.0]]
    remaining = n_points - 1
    ring = 1
    while remaining > 0:
        r = ring * 0.19
        if r > 0.95:
            r = 0.95
        count = min(remaining, max(1, int(2 * np.pi * r / 0.19)))
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.pi / count
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
        remaining -= count
        ring += 1
    return np.array(pts[:n_points])


def _random_init(n_points, seed=0):
    """Random initialization inside the unit disk."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(n_points):
        r = np.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * np.pi)
        pts.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(pts)


def _optimize_single(pts0, beta_start=20.0, beta_end=2000.0, max_iter=500, time_limit=15.0):
    """Run multi-stage annealing optimization on a single initial configuration."""
    start_time = time.time()
    pts = pts0.copy()
    
    # Multi-stage beta annealing
    n_stages = 6
    betas = np.geomspace(beta_start, beta_end, n_stages)
    
    for stage_beta in betas:
        if time.time() - start_time > time_limit:
            break
        try:
            result = minimize(
                _objective,
                pts,
                args=(stage_beta,),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'gtol': 1e-8, 'maxls': 50}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                pts = result.x
        except Exception:
            pass
    
    return pts


def _force_directed_precondition(pts, iterations=200, repulsion=1.0, attraction=0.1):
    """Simple force-directed relaxation to spread points out."""
    pts = pts.copy()
    n = len(pts)
    for it in range(iterations):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.sqrt(np.sum(d * d))
                if dist < 1e-10:
                    dist = 1e-10
                    d = np.random.randn(2) * 1e-10
                # Repulsive force
                f = repulsion / (dist * dist)
                forces[i] += f * d / dist
                forces[j] -= f * d / dist
            # Attraction toward center (to keep them in disk)
            r = np.sqrt(np.sum(pts[i] * pts[i]))
            if r > 0.1:
                forces[i] -= attraction * pts[i] / r
        # Damped update
        step = 0.01 / (1.0 + it * 0.01)
        pts += step * forces
        # Project back into disk
        radii = np.sqrt(np.sum(pts * pts, axis=1))
        mask = radii > 0.99
        if np.any(mask):
            pts[mask] *= 0.99 / radii[mask]
    return pts


def construct():
    """Construct 50 circle centers in the unit disk maximizing the common radius."""
    global DEADLINE
    start_time = time.time()
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_hex_lattice_init(N))
    inits.append(_ring_init(N))
    for seed in range(8):
        inits.append(_random_init(N, seed=seed * 42 + 7))
    
    best_pts = inits[0].copy()
    best_score = _score(best_pts)
    
    # Force-directed preconditioning for each init
    precond_inits = []
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        precond = _force_directed_precondition(init, iterations=100)
        precond_inits.append(precond)
    
    # Optimize each pre-conditioned configuration
    for i, init in enumerate(precond_inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        remaining = DEADLINE - (time.time() - start_time)
        time_limit = min(12.0, remaining - 2.0)
        if time_limit < 3.0:
            break
        try:
            result_pts = _optimize_single(init, beta_start=20.0, beta_end=5000.0,
                                           max_iter=300, time_limit=time_limit)
            s = _score(result_pts)
            if s > best_score:
                best_score = s
                best_pts = result_pts.copy()
        except Exception:
            pass
    
    # Final local polish on the best result
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                _objective,
                best_pts,
                args=(10000.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-10, 'maxls': 30}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                s = _score(result.x)
                if s > best_score:
                    best_pts = result.x.copy()
                    best_score = s
        except Exception:
            pass
    
    # Safety: ensure all points are strictly inside the unit disk
    radii = np.sqrt(np.sum(best_pts * best_pts, axis=1))
    mask = radii >= 1.0
    if np.any(mask):
        best_pts[mask] *= (1.0 - 1e-6) / radii[mask]
    
    # Final check
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback to hex lattice
        best_pts = _hex_lattice_init(N)
    
    return best_pts
