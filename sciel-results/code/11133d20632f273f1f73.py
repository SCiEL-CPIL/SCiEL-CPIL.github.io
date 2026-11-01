"""Improved packer: force-directed pre-conditioning + soft-min annealing + high-precision polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SEED = 42
np.random.seed(SEED)

# Deadline guard
_START = time.time()
_DEADLINE = 140.0  # seconds; evaluator kills at 150s


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _score(points):
    """Compute the common radius for a set of points."""
    # Boundary constraint
    dists_from_origin = np.linalg.norm(points, axis=1)
    boundary_term = 1.0 - dists_from_origin
    if np.any(boundary_term < -1e-12):
        return 0.0
    # Pairwise distances
    n = len(points)
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Exclude diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_pair, np.min(boundary_term))
    return max(r, 0.0)


def _soft_min_loss(points, beta):
    """
    Smooth surrogate: -softmin of all constraints (pairwise dist/2 and boundary 1-|ci|).
    We maximize the min, so minimize -min = soft-max of negatives.
    Using log-sum-exp for stability.
    """
    n = len(points)
    # Boundary terms: 1 - |c_i|
    dists_origin = np.linalg.norm(points, axis=1)
    boundary = 1.0 - dists_origin  # shape (n,)
    
    # Pairwise distances / 2
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2) / 2.0
    np.fill_diagonal(dists, np.inf)
    pairwise = dists[~np.eye(n, dtype=bool)]  # flatten off-diagonal
    
    # Combine all terms
    all_terms = np.concatenate([boundary, pairwise])
    
    # softmin = -logsumexp(-beta * terms) / beta
    # For numerical stability, shift by max
    shifted = -beta * all_terms
    max_shifted = np.max(shifted)
    if not np.isfinite(max_shifted):
        return 0.0, 0.0
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    softmin_val = -log_sum / beta
    
    # We want to maximize softmin, so loss = -softmin
    loss = -softmin_val
    return loss, softmin_val


def _hex_lattice_init(n):
    """Generate n points from hexagonal lattice within unit disk."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            # Sort by distance from center and take first n
            idx = np.argsort(radius[radius <= 1.0 - spacing / 2.0], kind='stable')
            return points[idx[:n]]
        spacing *= 0.98


def _force_directed_precondition(points, iters=200):
    """
    Apply force-directed relaxation: repulsive forces between circles and attractive to center.
    This helps escape poor local minima before optimization.
    """
    n = len(points)
    points = points.copy()
    # Estimate initial radius
    r_est = _score(points)
    if r_est <= 0:
        r_est = 0.1
    
    dt = 0.001
    for it in range(iters):
        if _remaining() < 5.0:
            break
        forces = np.zeros_like(points)
        # Repulsive forces between pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([1e-10, 0.0])
                    dist = 1e-10
                # Force proportional to overlap
                if dist < 2 * r_est:
                    force_mag = (2 * r_est - dist) * 10.0
                    unit = diff / dist
                    forces[i] += force_mag * unit
                    forces[j] -= force_mag * unit
        # Attractive force to center (weak)
        for i in range(n):
            forces[i] -= 0.1 * points[i]
        # Update
        points += dt * forces
        # Project back inside unit disk
        norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if norms[i] > 0.99:
                points[i] *= 0.99 / norms[i]
    return points


def _optimize_from_init(init_points, beta_schedule, max_iter=500):
    """
    Optimize using L-BFGS-B on the soft-min loss with annealing beta.
    """
    n = len(init_points)
    points = init_points.copy()
    best_points = points.copy()
    best_r = _score(points)
    
    for beta in beta_schedule:
        if _remaining() < 10.0:
            break
        
        def loss_func(x):
            p = x.reshape(-1, 2)
            loss, _ = _soft_min_loss(p, beta)
            return loss
        
        # Bounds: keep points well inside unit disk
        bounds = [(-0.999, 0.999)] * (2 * n)
        
        result = minimize(loss_func, points.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'gtol': 1e-10})
        
        if result.fun < loss_func(points.flatten()):
            points = result.x.reshape(-1, 2)
            r = _score(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    return best_points, best_r


def construct():
    if _elapsed() > 10:
        # Fallback: return hex lattice
        return _hex_lattice_init(N)
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    hex_pts = _hex_lattice_init(N)
    inits.append(hex_pts)
    
    # 2. Random with force-directed preconditioning
    for trial in range(5):
        if _remaining() < 20.0:
            break
        # Random points inside disk
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.95
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = _force_directed_precondition(pts, iters=100)
        inits.append(pts)
    
    # 3. Ring-based initialization
    if _remaining() > 20.0:
        # Multiple rings
        ring_radii = np.linspace(0.1, 0.9, 5)
        pts_ring = []
        for r_ring in ring_radii:
            n_ring = int(N / len(ring_radii))
            angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            pts_ring.extend([r_ring * np.cos(a), r_ring * np.sin(a)] for a in angles)
        pts_ring = np.array(pts_ring[:N])
        if len(pts_ring) == N:
            inits.append(pts_ring)
    
    # Beta schedule for annealing
    beta_schedule = [50, 100, 500, 2000, 10000, 50000, 100000]
    
    best_points = None
    best_r = 0.0
    
    for init in inits:
        if _remaining() < 15.0:
            break
        pts, r = _optimize_from_init(init, beta_schedule, max_iter=300)
        if r > best_r:
            best_r = r
            best_points = pts
    
    # Final polish: high-precision local optimization
    if best_points is not None and _remaining() > 5.0:
        # Use very high beta for precise convergence
        final_beta = 100000
        
        def loss_func(x):
            p = x.reshape(-1, 2)
            loss, _ = _soft_min_loss(p, final_beta)
            return loss
        
        bounds = [(-0.999, 0.999)] * (2 * N)
        result = minimize(loss_func, best_points.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 1000, 'gtol': 1e-12})
        
        if result.fun < loss_func(best_points.flatten()):
            polished = result.x.reshape(-1, 2)
            r_polished = _score(polished)
            if r_polished > best_r:
                best_points = polished
                best_r = r_polished
    
    if best_points is None:
        return _hex_lattice_init(N)
    
    return best_points
