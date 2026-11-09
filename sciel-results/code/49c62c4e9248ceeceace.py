"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers."""
    n = len(pts)
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary_dist)
    min_pair = pdist(pts) / 2.0
    min_pair = np.min(min_pair)
    return min(min_boundary, min_pair)


def _dynamics_simulate(pts, deadline, n_steps=500, dt=0.01, k_rep=10.0, k_boundary=5.0, cooling=0.999):
    """
    Rigid-body dynamics simulation with repulsive and boundary forces.
    Each circle is a particle; forces push circles apart and toward the boundary shell.
    """
    pts = pts.copy()
    n = len(pts)
    velocities = np.zeros_like(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Repulsive forces between pairs (inverse square law, capped)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    # Random small push
                    forces[i] += np.random.randn(2) * 0.1
                    forces[j] -= np.random.randn(2) * 0.1
                else:
                    # Repulsive force: k_rep / dist^2 in direction of diff
                    force_mag = k_rep / (dist * dist)
                    # Cap the force to avoid explosions
                    force_mag = min(force_mag, 10.0)
                    unit_diff = diff / dist
                    forces[i] += force_mag * unit_diff
                    forces[j] -= force_mag * unit_diff
        
        # Boundary force: pull toward shell at distance 1 - r_est
        # Estimate current radius
        r_est = _compute_radius(pts)
        shell_dist = 1.0 - r_est
        
        for i in range(n):
            dist_to_center = np.linalg.norm(pts[i])
            if dist_to_center > 1e-6:
                # Force toward the shell
                radial_dir = pts[i] / dist_to_center
                # If too close to center, push outward; if too far, push inward
                # We want |pts[i]| ≈ shell_dist
                delta_r = shell_dist - dist_to_center
                force_mag = k_boundary * delta_r
                forces[i] += force_mag * radial_dir
        
        # Velocity Verlet integration
        # v(t + dt/2) = v(t) + a(t) * dt/2
        velocities += forces * dt / 2.0
        
        # x(t + dt) = x(t) + v(t + dt/2) * dt
        pts += velocities * dt
        
        # Compute new forces at new positions
        new_forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    new_forces[i] += np.random.randn(2) * 0.1
                    new_forces[j] -= np.random.randn(2) * 0.1
                else:
                    force_mag = min(k_rep / (dist * dist), 10.0)
                    unit_diff = diff / dist
                    new_forces[i] += force_mag * unit_diff
                    new_forces[j] -= force_mag * unit_diff
        
        for i in range(n):
            dist_to_center = np.linalg.norm(pts[i])
            if dist_to_center > 1e-6:
                radial_dir = pts[i] / dist_to_center
                delta_r = shell_dist - dist_to_center
                force_mag = k_boundary * delta_r
                new_forces[i] += force_mag * radial_dir
        
        # v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        velocities += new_forces * dt / 2.0
        
        # Damping
        velocities *= cooling
        
        # Track best
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    return best_pts, best_r


def _soft_min_radius(pts, beta=100.0):
    """
    Compute a soft-min approximation of the radius using log-sum-exp.
    """
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    pair_dists = pdist(pts) / 2.0
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
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
    Compute the gradient of the soft-min radius.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    pair_dists = pdist(pts) / 2.0
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    min_val = np.min(all_dists)
    if min_val < -100:
        return grad
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return grad
    
    weights = exp_vals / sum_exp
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_grad = -pts / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * boundary_grad
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    pair_weights = weights[n:]
    pair_grads = pair_weights[:, np.newaxis] * (diffs / safe_dists[:, np.newaxis])
    
    np.add.at(grad, i_idx, pair_grads)
    np.add.at(grad, j_idx, -pair_grads)
    
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
    r1, n1 = 0.3, 6
    for i in range(n1):
        angle = 2 * np.pi * i / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    r2, n2 = 0.65, 12
    for i in range(n2):
        angle = 2 * np.pi * i / n2 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    r3, n3 = 0.85, 18
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


def _pso_optimize(initial_pts, deadline, n_particles=20, max_iter=50):
    """
    Particle Swarm Optimization for circle packing.
    Each particle is a full configuration of N points.
    """
    if time.time() > deadline - 10:
        return initial_pts, _compute_radius(initial_pts)
    
    # Initialize particles
    particles = []
    velocities = []
    
    # First particle is the initial point
    particles.append(initial_pts.copy())
    velocities.append(np.zeros_like(initial_pts))
    
    # Other particles are random perturbations
    np.random.seed(42)
    for _ in range(n_particles - 1):
        perturbation = np.random.randn(*initial_pts.shape) * 0.1
        p = initial_pts + perturbation
        particles.append(p)
        velocities.append(np.random.randn(*initial_pts.shape) * 0.05)
    
    # Initialize best positions
    p_best = particles.copy()
    p_best_r = [_compute_radius(p) for p in p_best]
    
    g_best_idx = np.argmax(p_best_r)
    g_best = particles[g_best_idx].copy()
    g_best_r = p_best_r[g_best_idx]
    
    # PSO parameters
    w = 0.7  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Update particles
        for i in range(n_particles):
            if time.time() > deadline - 5:
                break
            
            r1 = np.random.rand(*particles[i].shape)
            r2 = np.random.rand(*particles[i].shape)
            
            # Velocity update
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (p_best[i] - particles[i]) +
                           c2 * r2 * (g_best - particles[i]))
            
            # Position update
            particles[i] = particles[i] + velocities[i]
            
            # Evaluate
            r = _compute_radius(particles[i])
            
            # Update personal best
            if r > p_best_r[i]:
                p_best[i] = particles[i].copy()
                p_best_r[i] = r
                
                # Update global best
                if r > g_best_r:
                    g_best = particles[i].copy()
                    g_best_r = r
        
        # Adaptive inertia weight
        w = 0.9 - 0.4 * it / max_iter
    
    return g_best, g_best_r


def _optimize_from_init(init_pts, beta_start=50.0, beta_end=500.0, n_anneal=10, deadline=None):
    """
    Optimize from an initial configuration using annealing.
    """
    if deadline is not None and time.time() > deadline:
        return init_pts, _compute_radius(init_pts)
    
    best_pts = init_pts.copy()
    best_r = _compute_radius(init_pts)
    
    n_anneal = max(1, min(n_anneal, 20))
    
    for step in range(n_anneal):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        t = step / max(1, n_anneal - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        def objective(x, b=beta):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, b)
        
        def gradient(x, b=beta):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, b)
        
        x0 = best_pts.flatten()
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < objective(x0):
                new_pts = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    return best_pts, best_r


def _nelder_mead_polish(pts, deadline=None):
    """Polish with a local search that pushes overlapping circles apart."""
    if deadline is not None and time.time() > deadline - 5:
        return pts
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    for iteration in range(100):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        moved = False
        n = len(best_pts)
        
        r = _compute_radius(best_pts)
        
        beta = 1000.0
        grad = _soft_min_radius_grad(best_pts, beta)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad > 1e-10:
            step_size = 0.005
            new_pts = best_pts + step_size * grad
            new_r = _compute_radius(new_pts)
            
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
                moved = True
        
        if not moved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate diverse initializations
    inits = []
    
    for spacing in [0.2, 0.22, 0.24, 0.26]:
        pts = _hex_init(spacing)
        if len(pts) >= N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii, kind='stable')[:N]
            inits.append(pts[idx])
        elif len(pts) > 0:
            inits.append(pts)
    
    ring_pts = _ring_init()
    if len(ring_pts) >= N:
        inits.append(ring_pts[:N])
    elif len(ring_pts) > 0:
        inits.append(ring_pts)
    
    for seed in [42, 123, 456]:
        if time.time() > deadline - 20:
            break
        inits.append(_random_init(seed))
    
    valid_inits = []
    for pts in inits:
        if len(pts) == N:
            r = _compute_radius(pts)
            if r > 0:
                valid_inits.append((pts, r))
    
    if not valid_inits:
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
    
    valid_inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = valid_inits[0][0]
    best_r = valid_inits[0][1]
    
    # Phase 1: Dynamics simulation from best initializations
    for i, (init_pts, init_r) in enumerate(valid_inits[:4]):
        if time.time() > deadline - 50:
            break
        
        dyn_pts, dyn_r = _dynamics_simulate(init_pts, deadline, n_steps=300, dt=0.008, k_rep=8.0, k_boundary=4.0)
        if dyn_r > best_r:
            best_pts = dyn_pts
            best_r = dyn_r
    
    # Phase 2: PSO exploration
    for i, (init_pts, init_r) in enumerate(valid_inits[:3]):
        if time.time() > deadline - 30:
            break
        
        pso_pts, pso_r = _pso_optimize(init_pts, deadline, n_particles=12, max_iter=25)
        if pso_r > best_r:
            best_pts = pso_pts
            best_r = pso_r
    
    # Phase 3: Gradient-based refinement
    for i, (init_pts, init_r) in enumerate(valid_inits[:2]):
        if time.time() > deadline - 15:
            break
        
        start_pts = best_pts if best_r > init_r else init_pts
        
        optimized_pts, optimized_r = _optimize_from_init(
            start_pts, 
            beta_start=50.0, 
            beta_end=500.0, 
            n_anneal=8,
            deadline=deadline
        )
        
        if optimized_r > best_r:
            best_pts = optimized_pts
            best_r = optimized_r
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        def final_objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, beta=1000.0)
        
        def final_gradient(x):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, beta=1000.0)
        
        try:
            res = minimize(final_objective, best_pts.flatten(), jac=final_gradient, 
                          method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < final_objective(best_pts.flatten()):
                new_pts = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    # Final local search polish
    if time.time() < deadline - 1:
        polished = _nelder_mead_polish(best_pts, deadline)
        r = _compute_radius(polished)
        if r > best_r:
            best_pts = polished
            best_r = r
    
    return best_pts
# EVOLVE-BLOCK-END
