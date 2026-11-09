"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers using vectorized ops."""
    n = len(pts)
    # Distance to boundary
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances using cdist (vectorized)
    dist_matrix = cdist(pts, pts)
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


def _hex_init(spacing):
    """Generate hexagonal lattice points."""
    k = int(1.0 / spacing) + 2
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append([x, y])
    points = np.array(rows)
    radius = np.linalg.norm(points, axis=1)
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    return points


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
    
    return np.array(points)


def _random_init(seed):
    """Generate random initialization."""
    np.random.seed(seed)
    points = []
    for _ in range(N):
        r = np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)


def _structured_init():
    """Generate a structured initialization based on known packing patterns."""
    # Use a pattern inspired by known good packings:
    # Center + rings with specific counts
    points = [[0.0, 0.0]]
    
    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 circles
    r2 = 0.55
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 circles
    r3 = 0.82
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # That's 1 + 6 + 12 + 18 = 37, need 13 more
    # Add a partial ring 4
    r4 = 0.95
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)


def _optimize_from_init(init_pts, deadline=None):
    """
    Optimize from an initial configuration using multi-stage annealing with L-BFGS-B.
    """
    if deadline is not None and time.time() > deadline:
        return init_pts, _compute_radius(init_pts)
    
    best_pts = init_pts.copy()
    best_r = _compute_radius(init_pts)
    
    # Multi-stage beta annealing
    betas = [30, 60, 120, 250, 500]
    
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5:
            break
        
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, beta)
        
        def gradient(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, beta)
        
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
    
    return best_pts, best_r


def _nelder_mead_polish(pts, deadline=None):
    """Polish with Nelder-Mead for local refinement."""
    if deadline is not None and time.time() > deadline - 3:
        return pts
    
    def objective(x):
        p = x.reshape(-1, 2)
        return -_compute_radius(p)
    
    try:
        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
        new_pts = res.x.reshape(-1, 2)
        new_r = _compute_radius(new_pts)
        old_r = _compute_radius(pts)
        if new_r > old_r:
            return new_pts
    except Exception:
        pass
    
    return pts


def _ps_optimize(init_pts, deadline=None, n_particles=20, max_iter=300, w=0.8, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization for circle packing.
    Each particle is a full configuration of 50 centers.
    """
    if deadline is not None and time.time() > deadline:
        return init_pts, _compute_radius(init_pts)
    
    dim = N * 2
    
    # Initialize particles
    particles = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    
    # First particle is the initial point
    particles[0] = init_pts.flatten()
    
    # Other particles are random perturbations of the initial point
    for i in range(1, n_particles):
        particles[i] = init_pts.flatten() + np.random.randn(dim) * 0.1
    
    # Evaluate initial fitness
    pbests = np.zeros((n_particles, dim))
    pbest_fitness = np.zeros(n_particles)
    gbest = init_pts.flatten().copy()
    gbest_fitness = -np.inf
    
    for i in range(n_particles):
        pts = particles[i].reshape(-1, 2)
        r = _compute_radius(pts)
        pbests[i] = particles[i]
        pbest_fitness[i] = r
        if r > gbest_fitness:
            gbest_fitness = r
            gbest = particles[i].copy()
    
    # PSO loop
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (pbests[i] - particles[i]) + 
                           c2 * r2 * (gbest - particles[i]))
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Evaluate new position
            pts = particles[i].reshape(-1, 2)
            r = _compute_radius(pts)
            
            # Update pbest
            if r > pbest_fitness[i]:
                pbest_fitness[i] = r
                pbests[i] = particles[i]
                
                # Update gbest
                if r > gbest_fitness:
                    gbest_fitness = r
                    gbest = particles[i].copy()
    
    return gbest.reshape(-1, 2), gbest_fitness


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal lattices with different spacings
    for spacing in [0.2, 0.22, 0.24, 0.26]:
        pts = _hex_init(spacing)
        if len(pts) >= N:
            # Take the N closest to center
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii, kind='stable')[:N]
            inits.append(pts[idx])
        elif len(pts) > 0:
            inits.append(pts)
    
    # Ring initialization
    ring_pts = _ring_init()
    if len(ring_pts) >= N:
        inits.append(ring_pts[:N])
    elif len(ring_pts) > 0:
        inits.append(ring_pts)
    
    # Structured initialization
    structured_pts = _structured_init()
    if len(structured_pts) == N:
        inits.append(structured_pts)
    
    # Random initializations
    for seed in [42, 123, 456]:
        if time.time() > deadline - 20:
            break
        inits.append(_random_init(seed))
    
    # Filter valid inits
    valid_inits = []
    for pts in inits:
        if len(pts) == N:
            r = _compute_radius(pts)
            if r > 0:
                valid_inits.append((pts, r))
    
    if not valid_inits:
        # Fallback: simple hex grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    # Sort by initial radius
    valid_inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = valid_inits[0][0]
    best_r = valid_inits[0][1]
    
    # First pass: gradient-based optimization from each initialization
    for i, (init_pts, init_r) in enumerate(valid_inits):
        if time.time() > deadline - 30:
            break
        
        # Skip if already very good
        if init_r > 0.125:
            # Polish directly
            polished = _nelder_mead_polish(init_pts, deadline)
            r = _compute_radius(polished)
            if r > best_r:
                best_pts = polished
                best_r = r
            continue
        
        # Annealing optimization
        optimized_pts, optimized_r = _optimize_from_init(
            init_pts, 
            deadline=deadline
        )
        
        if optimized_r > best_r:
            best_pts = optimized_pts
            best_r = optimized_r
        
        # Final polish
        if time.time() < deadline - 20:
            polished = _nelder_mead_polish(best_pts, deadline)
            r = _compute_radius(polished)
            if r > best_r:
                best_pts = polished
                best_r = r
    
    # Second pass: PSO from the best gradient-based result
    if time.time() < deadline - 40:
        pso_pts, pso_r = _ps_optimize(
            best_pts, 
            deadline=deadline,
            n_particles=15,
            max_iter=200
        )
        if pso_r > best_r:
            best_pts = pso_pts
            best_r = pso_r
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        # Use higher beta for final L-BFGS pass
        def final_objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, beta=500.0)
        
        def final_gradient(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, beta=500.0)
        
        try:
            res = minimize(final_objective, best_pts.flatten(), jac=final_gradient, 
                          method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < final_objective(best_pts.flatten()):
                new_pts = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    # Nelder-Mead final polish
    if time.time() < deadline - 2:
        polished = _nelder_mead_polish(best_pts, deadline)
        r = _compute_radius(polished)
        if r > best_r:
            best_pts = polished
            best_r = r
    
    return best_pts
# EVOLVE-BLOCK-END
