"""Packing 50 equal circles in a unit circle using force-directed graph optimization."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the maximum common radius for given points."""
    if len(points) < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(len(points), k=1)
    min_pair = np.min(dists[iu]) / 2.0
    norms = np.hypot(points[:, 0], points[:, 1])
    min_boundary = 1.0 - np.max(norms)
    return min(min_pair, min_boundary)


def _ring_initialization(ring_counts=None, ring_radii=None):
    """Generate ring-structured initialization."""
    if ring_counts is None:
        ring_counts = [1, 6, 12, 18, 13]
    if ring_radii is None:
        ring_radii = [0.0, 0.22, 0.45, 0.68, 0.88]
    
    points = []
    for ring_idx, n in enumerate(ring_counts):
        r = ring_radii[ring_idx]
        if n == 1:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points)


def _hexagonal_initialization():
    """Generate hexagonal lattice initialization."""
    points = []
    # Hexagonal lattice with spacing ~0.25
    spacing = 0.25
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    # Generate points in a hexagonal pattern
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if np.sqrt(x*x + y*y) < 0.95:
                points.append([x, y])
    
    # Sort by distance from center and take first 50
    points = np.array(points)
    if len(points) > N:
        norms = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(norms)[:N]
        points = points[idx]
    
    # If we have fewer than 50, pad with ring structure
    if len(points) < N:
        ring_pts = _ring_initialization()
        # Add remaining from ring
        existing_norms = np.hypot(points[:, 0], points[:, 1])
        for p in ring_pts:
            if len(points) >= N:
                break
            if not any(np.allclose(p, ep, atol=0.01) for ep in points):
                points = np.vstack([points, [p]])
    
    return points[:N]


def _force_directed_energy(points, target_r, k_rep=10.0, k_attr=1.0):
    """Compute force-directed energy: repulsion for close pairs, attraction for contacts."""
    points = points.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    energy = 0.0
    # Repulsion for pairs closer than 2*target_r
    close_mask = pair_dists < 2 * target_r
    if np.any(close_mask):
        close_dists = pair_dists[close_mask]
        overlaps = 2 * target_r - close_dists
        energy += k_rep * np.sum(overlaps ** 2)
    
    # Boundary violation
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_viol = np.maximum(0, norms - (1.0 - target_r))
    energy += k_rep * np.sum(boundary_viol ** 2)
    
    return energy


def _force_directed_grad(points_flat, target_r, k_rep=10.0):
    """Gradient of force-directed energy."""
    points = points_flat.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    
    grad = np.zeros_like(points_flat)
    
    # Pair repulsion gradients
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        d = dists[i, j]
        if d < 2 * target_r and d > 1e-15:
            direction = (points[i] - points[j]) / d
            overlap = 2 * target_r - d
            force = 2 * k_rep * overlap
            grad[2*i] += force * direction[0]
            grad[2*i+1] += force * direction[1]
            grad[2*j] -= force * direction[0]
            grad[2*j+1] -= force * direction[1]
    
    # Boundary gradients
    norms = np.hypot(points[:, 0], points[:, 1])
    for i in range(N):
        if norms[i] > 1.0 - target_r and norms[i] > 1e-15:
            direction = points[i] / norms[i]
            force = 2 * k_rep * (norms[i] - (1.0 - target_r))
            grad[2*i] += force * direction[0]
            grad[2*i+1] += force * direction[1]
    
    return grad


def _soft_min_objective(points_flat, beta):
    """Soft-min objective using log-sum-exp for smooth minimization."""
    points = points_flat.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - norms
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft minimum via log-sum-exp
    scaled = -beta * all_dists
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_scaled)
    soft_min = -(max_val + np.log(sum_exp)) / beta
    
    return -soft_min


def _soft_min_grad(points_flat, beta):
    """Gradient of soft-min objective."""
    points = points_flat.reshape(N, 2)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_dists = 1.0 - norms
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    scaled = -beta * all_dists
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    grad = np.zeros_like(points_flat)
    
    # Pair distance gradients
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = weights[idx]
        d = pair_dists[idx]
        if d > 1e-15:
            direction = (points[i] - points[j]) / d
            grad[2*i] += w * direction[0]
            grad[2*i+1] += w * direction[1]
            grad[2*j] -= w * direction[0]
            grad[2*j+1] -= w * direction[1]
    
    # Boundary distance gradients
    for idx in range(N):
        w = weights[N + idx]
        norm = norms[idx]
        if norm > 1e-15:
            grad[2*idx] -= w * points[idx, 0] / norm
            grad[2*idx+1] -= w * points[idx, 1] / norm
    
    return grad


def _local_polish(points, max_iters=200):
    """Local polish: push apart overlapping pairs and pull inward if outside."""
    points = points.copy()
    n = len(points)
    
    for iteration in range(max_iters):
        if time.time() > _deadline:
            break
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        norms = np.hypot(points[:, 0], points[:, 1])
        
        iu = np.triu_indices(n, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        min_boundary = 1.0 - np.max(norms)
        current_r = min(min_pair, min_boundary)
        
        improved = False
        
        # Push apart close pairs
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dists[i, j]
            if d < 2 * current_r - 1e-12 and d > 1e-15:
                direction = (points[i] - points[j]) / d
                overlap = (2 * current_r - d) * 0.5
                move = overlap * 0.1
                points[i] += direction * move
                points[j] -= direction * move
                improved = True
        
        # Pull inward if too close to boundary
        for i in range(n):
            norm_i = norms[i]
            if norm_i > 1.0 - current_r + 1e-12 and norm_i > 1e-15:
                direction = points[i] / norm_i
                overlap = (norm_i - (1.0 - current_r)) * 0.5
                move = overlap * 0.1
                points[i] -= direction * move
                improved = True
        
        if not improved:
            break
    
    return points


def _multi_start_optimize(initial_points, deadline):
    """Multi-start optimization with soft-min and force-directed methods."""
    global _deadline
    _deadline = deadline
    
    best_points = initial_points.copy()
    best_score = _compute_score(best_points)
    
    rng = np.random.default_rng(123)
    
    # Phase 1: Force-directed relaxation with target radius
    for target_r in [0.12, 0.123, 0.125, 0.126, 0.127, 0.128]:
        if time.time() > deadline - 30.0:
            break
        
        x0 = best_points.flatten().copy()
        result = minimize(
            lambda x: _force_directed_energy(x, target_r),
            x0,
            jac=lambda x: _force_directed_grad(x, target_r),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result.success:
            opt_points = result.x.reshape(N, 2)
            if np.all(np.isfinite(opt_points)):
                norms = np.hypot(opt_points[:, 0], opt_points[:, 1])
                if np.all(norms <= 1.0):
                    score = _compute_score(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
    
    # Phase 2: Soft-min optimization with increasing beta
    for restart in range(10):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            x0 = best_points.flatten().copy()
        else:
            noise_scale = 0.03 * np.exp(-0.3 * restart)
            x0 = best_points.flatten().copy()
            x0 += rng.normal(0, noise_scale, size=2*N)
            pts = x0.reshape(N, 2)
            norms = np.hypot(pts[:, 0], pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
            x0 = pts.flatten()
        
        for beta in [50.0, 100.0, 200.0, 400.0, 800.0]:
            if time.time() > deadline - 10.0:
                break
            
            result = minimize(
                _soft_min_objective,
                x0,
                jac=_soft_min_grad,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 25}
            )
            
            if result.success or result.fun < 1e10:
                opt_points = result.x.reshape(N, 2)
                if np.all(np.isfinite(opt_points)):
                    norms = np.hypot(opt_points[:, 0], opt_points[:, 1])
                    if np.all(norms <= 1.0):
                        score = _compute_score(opt_points)
                        if score > best_score:
                            best_score = score
                            best_points = opt_points.copy()
                            x0 = opt_points.flatten().copy()
    
    # Phase 3: Final local polish
    if time.time() < deadline - 5.0:
        polished = _local_polish(best_points, max_iters=300)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    return best_points


def construct():
    global _deadline
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s kill
    _deadline = deadline
    
    # Generate multiple initial configurations
    initializations = []
    
    # Ring structures with different radii
    ring_configs = [
        [0.0, 0.22, 0.45, 0.68, 0.88],
        [0.0, 0.20, 0.42, 0.65, 0.85],
        [0.0, 0.25, 0.50, 0.72, 0.90],
        [0.0, 0.18, 0.38, 0.60, 0.82],
        [0.0, 0.23, 0.47, 0.70, 0.89],
    ]
    for radii in ring_configs:
        initializations.append(_ring_initialization(ring_radii=radii))
    
    # Hexagonal lattice
    initializations.append(_hexagonal_initialization())
    
    # Select best initial configuration
    best_init = None
    best_init_score = -1.0
    for init in initializations:
        score = _compute_score(init)
        if score > best_init_score:
            best_init_score = score
            best_init = init
    
    # Run optimization from best initialization
    result_points = _multi_start_optimize(best_init, deadline)
    
    return result_points
# EVOLVE-BLOCK-END
