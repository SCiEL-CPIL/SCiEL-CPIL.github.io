import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers using vectorized ops."""
    n = len(pts)
    # Distance to boundary
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances using cdist (vectorized)
    dist_matrix = cdist(pts, pts)
    # Get upper triangle excluding diagonal
    iu = np.triu_indices(n, k=1)
    min_pair = np.min(dist_matrix[iu]) / 2.0
    
    return min(min_boundary, min_pair)


def _soft_min_radius(pts, beta=100.0):
    """
    Compute a soft-min approximation of the radius using log-sum-exp.
    Vectorized version.
    """
    n = len(pts)
    
    # Boundary distances
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise distances (half) using cdist
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu] / 2.0
    
    # Combine all constraints into one array
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_dists)
    if min_val < -100:
        return min_val
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return min_val
    
    soft_min = min_val - np.log(sum_exp) / beta
    
    return soft_min


def _soft_min_radius_grad(pts, beta=100.0):
    """
    Compute the gradient of the soft-min radius. Vectorized.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    
    # Pairwise distances using cdist
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu] / 2.0
    
    # Combine all constraints
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    min_val = np.min(all_dists)
    if min_val < -100:
        return grad
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return grad
    
    weights = exp_vals / sum_exp
    
    # Boundary terms - vectorized
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_grad = weights[:n, np.newaxis] * (-pts / safe_norms[:, np.newaxis])
    grad += boundary_grad
    
    # Pair terms - vectorized
    i_idx, j_idx = iu
    diff = pts[i_idx] - pts[j_idx]
    norm_diff = np.linalg.norm(diff, axis=1)
    safe_norm_diff = np.where(norm_diff > 1e-10, norm_diff, 1.0)
    pair_weights = weights[n:]
    
    # For each pair (i,j), grad[i] += w * (diff/|diff|), grad[j] += w * (-diff/|diff|)
    unit_diff = diff / safe_norm_diff[:, np.newaxis]
    
    # Accumulate using np.add.at for repeated indices
    np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_diff)
    np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_diff)
    
    return grad


def _ring_init():
    """Generate ring-based initialization matching known optimal structure."""
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 circles
    r1 = 0.251
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 circles
    r2 = 0.552
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 circles
    r3 = 0.782
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 circles (outer)
    r4 = 0.935
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)


def _hex_init():
    """Generate hexagonal lattice points."""
    # Approximate spacing for N circles
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    # Sort by distance to center, take closest N
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _random_init():
    """Generate a random initialization inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    x = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return x


def _optimize_gradient(pts, deadline):
    """
    Refine a configuration using multi-stage annealing with L-BFGS-B.
    """
    if time.time() > deadline:
        return pts
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Multi-stage beta annealing
    betas = [30, 60, 120, 250, 500]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        def objective(x):
            p = x.reshape(-1, 2)
            return -_soft_min_radius(p, beta)
        
        def gradient(x):
            p = x.reshape(-1, 2)
            return -_soft_min_radius_grad(p, beta)
        
        x0 = best_pts.flatten()
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            
            new_pts = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
        except Exception:
            pass
    
    return best_pts


def _nelder_mead_polish(pts, deadline, max_iter=200):
    """
    Final polish using Nelder-Mead with tight tolerances.
    """
    if time.time() > deadline:
        return pts
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    def objective(x):
        p = x.reshape(-1, 2)
        return -_compute_radius(p)
    
    x0 = best_pts.flatten()
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        
        new_pts = res.x.reshape(-1, 2)
        new_r = _compute_radius(new_pts)
        if new_r > best_r:
            best_pts = new_pts
            best_r = new_r
    except Exception:
        pass
    
    return best_pts


def _pso_optimize(pts, deadline, n_particles=8, max_iter=40):
    """
    Particle Swarm Optimization to escape local minima.
    Each particle is a full configuration of N points.
    """
    if time.time() > deadline:
        return pts
    
    dim = N * 2
    
    # Initialize particles
    particles = np.zeros((n_particles, dim))
    particles[0] = pts.flatten()
    
    for i in range(1, n_particles):
        if i % 2 == 0:
            # Random init
            particles[i] = _random_init().flatten()
        else:
            # Perturbed current best
            particles[i] = pts.flatten() + np.random.normal(0, 0.03, dim)
        
        # Ensure inside circle (clamp)
        p = particles[i].reshape(-1, 2)
        norms = np.linalg.norm(p, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            p[mask] = p[mask] / norms[mask] * 0.99
    
    # Velocities
    velocities = np.zeros_like(particles)
    
    # Personal and Global Bests
    p_best = particles.copy()
    p_best_scores = np.array([_compute_radius(p.reshape(-1, 2)) for p in particles])
    
    g_best_idx = np.argmax(p_best_scores)
    g_best = p_best[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]
    
    w = 0.7  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    # PSO Loop
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
        
        # Update velocities and positions
        for i in range(n_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
            
            # Limit velocity magnitude to prevent divergence
            v_norm = np.linalg.norm(velocities[i])
            max_v = 0.05
            if v_norm > max_v:
                velocities[i] = velocities[i] / v_norm * max_v
            
            particles[i] = particles[i] + velocities[i]
            
            # Constrain to unit circle
            p = particles[i].reshape(-1, 2)
            norms = np.linalg.norm(p, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                p[mask] = p[mask] / norms[mask] * 0.99
            particles[i] = p.flatten()
            
            # Update personal best
            score = _compute_radius(particles[i].reshape(-1, 2))
            if score > p_best_scores[i]:
                p_best[i] = particles[i].copy()
                p_best_scores[i] = score
                
                # Update global best
                if score > g_best_score:
                    g_best = particles[i].copy()
                    g_best_score = score
    
    return g_best.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin
    
    best_x = _ring_init()
    best_r = _compute_radius(best_x)
    
    # List of initializations
    inits = [best_x, _hex_init(), _random_init(), _random_init()]
    
    # 1. Gradient-based refinement for each init
    for x_init in inits:
        if time.time() > deadline - 30:
            break
        x_opt = _optimize_gradient(x_init, deadline - 30)
        r_opt = _compute_radius(x_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt
    
    # 2. PSO to escape local minima (only if we have time)
    if time.time() < deadline - 25:
        x_pso = _pso_optimize(best_x, deadline - 15, n_particles=8, max_iter=40)
        r_pso = _compute_radius(x_pso)
        if r_pso > best_r:
            best_r = r_pso
            best_x = x_pso
    
    # 3. Final gradient refinement
    if time.time() < deadline - 10:
        x_ref = _optimize_gradient(best_x, deadline - 5)
        r_ref = _compute_radius(x_ref)
        if r_ref > best_r:
            best_r = r_ref
            best_x = x_ref
    
    # 4. Nelder-Mead polish for final precision
    if time.time() < deadline - 2:
        x_polish = _nelder_mead_polish(best_x, deadline - 1, max_iter=300)
        r_polish = _compute_radius(x_polish)
        if r_polish > best_r:
            best_r = r_polish
            best_x = x_polish
    
    return best_x
