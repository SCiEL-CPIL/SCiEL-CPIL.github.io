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
    
    # Boundary terms
    for i in range(n):
        w = weights[i]
        if norms[i] > 1e-10:
            grad[i] += w * (-pts[i] / norms[i])
    
    # Pair terms
    for k in range(len(iu[0])):
        w = weights[n + k]
        i, j = iu[0][k], iu[1][k]
        diff = pts[i] - pts[j]
        norm_diff = np.linalg.norm(diff)
        if norm_diff > 1e-10:
            grad[i] += w * (diff / norm_diff)
            grad[j] += w * (-diff / norm_diff)
    
    return grad


def _ring_init():
    """Generate ring-based initialization."""
    points = [[0.0, 0.0]]
    
    r1 = 0.3
    n1 = 6
    for i in range(n1):
        angle = 2 * np.pi * i / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    r2 = 0.65
    n2 = 12
    for i in range(n2):
        angle = 2 * np.pi * i / n2 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    r3 = 0.85
    n3 = 18
    for i in range(n3):
        angle = 2 * np.pi * i / n3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Need 50 total. Currently 1+6+12+18=37. Add 13 more in an outer ring.
    r4 = 0.95
    n4 = 13
    for i in range(n4):
        angle = 2 * np.pi * i / n4 + np.pi / 13
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
                          options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
            
            new_pts = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
        except Exception:
            pass
    
    return best_pts


def _pso_optimize(pts, deadline, n_particles=10, max_iter=50):
    """
    Particle Swarm Optimization to escape local minima.
    Each particle is a full configuration of N points.
    """
    if time.time() > deadline:
        return pts
    
    dim = N * 2
    
    # Initialize particles
    # Particle 0 is the input pts
    # Others are random perturbations or random inits
    particles = np.zeros((n_particles, dim))
    particles[0] = pts.flatten()
    
    for i in range(1, n_particles):
        if i % 2 == 0:
            # Random init
            particles[i] = _random_init().flatten()
        else:
            # Perturbed current best
            particles[i] = pts.flatten() + np.random.normal(0, 0.05, dim)
        
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
        if time.time() > deadline - 5:
            break
        
        # Update velocities and positions
        for i in range(n_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
            
            # Limit velocity magnitude to prevent divergence
            v_norm = np.linalg.norm(velocities[i])
            max_v = 0.1
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
    deadline = start_time + 110.0 # 10s safety margin
    
    best_x = _ring_init()
    best_r = _compute_radius(best_x)
    
    # List of initializations
    inits = [best_x, _hex_init(), _random_init(), _random_init()]
    
    # 1. Gradient-based refinement for each init
    for x_init in inits:
        if time.time() > deadline - 20:
            break
        x_opt = _optimize_gradient(x_init, deadline - 20)
        r_opt = _compute_radius(x_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_x = x_opt
    
    # 2. PSO to escape local minima
    # Use the best gradient result as the seed for PSO
    if time.time() < deadline - 10:
        x_pso = _pso_optimize(best_x, deadline - 5, n_particles=15, max_iter=60)
        r_pso = _compute_radius(x_pso)
        if r_pso > best_r:
            best_r = r_pso
            best_x = x_pso
            
            # Refine PSO result with gradient
            if time.time() < deadline - 2:
                x_ref = _optimize_gradient(best_x, deadline - 1)
                r_ref = _compute_radius(x_ref)
                if r_ref > best_r:
                    best_r = r_ref
                    best_x = x_ref
    
    return best_x
