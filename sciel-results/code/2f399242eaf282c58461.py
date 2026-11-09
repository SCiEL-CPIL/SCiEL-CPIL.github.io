"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s wall-clock kill


def _hex_grid_init(n):
    """Generate a hexagonal lattice initial packing, trimmed to n points."""
    best_pts = None
    best_score = 0.0
    for s in np.linspace(0.15, 0.35, 40):
        pts = []
        k = int(1.0 / s) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - s / 2.0) ** 2:
                    pts.append((x, y))
        if len(pts) >= n:
            pts_arr = np.array(pts)
            dists = np.hypot(pts_arr[:, 0], pts_arr[:, 1])
            idx = np.argsort(dists, kind='stable')[:n]
            candidate = pts_arr[idx]
            score = _score(candidate)
            if score > best_score:
                best_score = score
                best_pts = candidate
    if best_pts is not None:
        return best_pts
    return np.random.randn(n, 2) * 0.5


def _score(pts):
    """Compute the packing radius (minimum of min pair dist/2 and min distance to boundary)."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_boundary)


def _project_to_disk(pts):
    """Project points to stay within unit disk with small margin."""
    norms = np.hypot(pts[:, 0], pts[:, 1])
    outside = norms > 0.999
    if np.any(outside):
        pts[outside] = pts[outside] / norms[outside, np.newaxis] * 0.999
    return pts


def _relax(pts, max_iter=200, tol=1e-12):
    """
    Relax the packing using a force-based method.
    Returns improved pts.
    """
    pts = pts.copy()
    n = len(pts)
    r = _score(pts)
    if r <= 0:
        return pts
    
    lr = 0.01
    
    for _ in range(max_iter):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        norms = np.hypot(pts[:, 0], pts[:, 1])
        
        min_pair = np.min(dists)
        min_bound = np.min(1.0 - norms)
        current_r = min(min_pair / 2.0, min_bound)
        
        if current_r <= r * (1 + tol):
            break
            
        r = current_r
        
        forces = np.zeros_like(pts)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff = np.zeros_like(dists)
            mask = (dists < 2.0 * r) & (dists > 1e-12)
            coeff[mask] = (2.0 * r - dists[mask]) / dists[mask]
            
            forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            bound_violation = r - (1.0 - norms)
            active_bound = bound_violation > 0
            dir_bound = np.zeros_like(pts)
            mask_bound = norms > 1e-12
            dir_bound[mask_bound] = -pts[mask_bound] / norms[mask_bound, np.newaxis]
            
            forces[active_bound] += bound_violation[active_bound, np.newaxis] * dir_bound[active_bound]
            
            pts += lr * forces
            pts = _project_to_disk(pts)

    return pts


def _pso_optimize(pts_init, n_particles=20, max_iter=300, w=0.9, c1=1.5, c2=1.5, t0=None, deadline=None):
    """
    Particle Swarm Optimization for circle packing.
    Each particle is a full configuration of N circle centers.
    """
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = _DEADLINE
    
    n = N
    dim = n * 2  # flatten each configuration to 1D
    
    # Initialize particles
    particles = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    
    # Use hex grid as base, add small perturbations for diversity
    base_flat = pts_init.flatten()
    for i in range(n_particles):
        if i == 0:
            particles[i] = base_flat.copy()
        else:
            particles[i] = base_flat + np.random.randn(dim) * 0.03
            particles[i] = _project_to_disk(particles[i].reshape(n, 2)).flatten()
    
    # Evaluate initial particles
    p_best = np.zeros((n_particles, dim))
    p_best_scores = np.zeros(n_particles)
    g_best = np.zeros(dim)
    g_best_score = -np.inf
    
    for i in range(n_particles):
        pts = particles[i].reshape(n, 2)
        pts = _project_to_disk(pts)
        particles[i] = pts.flatten()
        score = _score(pts)
        p_best[i] = particles[i]
        p_best_scores[i] = score
        if score > g_best_score:
            g_best_score = score
            g_best = particles[i].copy()
    
    # PSO iteration
    rng = np.random.RandomState(42)
    for it in range(max_iter):
        if time.time() - t0 > deadline - 5:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = rng.rand(dim)
            r2 = rng.rand(dim)
            velocities[i] = w * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Project to disk
            pts = particles[i].reshape(n, 2)
            pts = _project_to_disk(pts)
            particles[i] = pts.flatten()
            
            # Evaluate
            score = _score(pts)
            
            # Update personal best
            if score > p_best_scores[i]:
                p_best_scores[i] = score
                p_best[i] = particles[i].copy()
                
                # Update global best
                if score > g_best_score:
                    g_best_score = score
                    g_best = particles[i].copy()
    
    return g_best.reshape(n, 2), g_best_score


def construct():
    t0 = time.time()
    n = N
    
    best_pts = None
    best_r = 0.0
    
    rng = np.random.RandomState(42)
    
    # 1. Hex grid initialization
    pts0 = _hex_grid_init(n)
    r0 = _score(pts0)
    if r0 > best_r:
        best_r = r0
        best_pts = pts0.copy()
        
    # 2. Relax hex grid
    if time.time() - t0 < _DEADLINE - 10:
        pts_relaxed = _relax(pts0, max_iter=100)
        r_relaxed = _score(pts_relaxed)
        if r_relaxed > best_r:
            best_r = r_relaxed
            best_pts = pts_relaxed.copy()
    
    # 3. PSO optimization
    if time.time() - t0 < _DEADLINE - 10:
        # Use the best configuration so far as PSO initialization
        pso_pts, pso_score = _pso_optimize(
            best_pts if best_pts is not None else pts0,
            n_particles=15,
            max_iter=200,
            w=0.9,
            c1=1.5,
            c2=1.5,
            t0=t0,
            deadline=_DEADLINE
        )
        if pso_score > best_r:
            best_r = pso_score
            best_pts = pso_pts.copy()
    
    # 4. Final relaxation polish
    if time.time() - t0 < _DEADLINE - 5 and best_pts is not None:
        pts_polished = _relax(best_pts, max_iter=50)
        r_polished = _score(pts_polished)
        if r_polished > best_r:
            best_r = r_polished
            best_pts = pts_polished.copy()
    
    # 5. Additional PSO run with more particles if time permits
    if time.time() - t0 < _DEADLINE - 15 and best_pts is not None:
        pso_pts2, pso_score2 = _pso_optimize(
            best_pts,
            n_particles=10,
            max_iter=150,
            w=0.85,
            c1=1.8,
            c2=1.2,
            t0=t0,
            deadline=_DEADLINE
        )
        if pso_score2 > best_r:
            best_r = pso_score2
            best_pts = pso_pts2.copy()
    
    # Final check
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_grid_init(n)
        
    return best_pts
