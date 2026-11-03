"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    energy = _log_energy(points)
    return energy, grads_tangent.flatten()


def _ps_optimize(points, start_time, deadline, max_iter=30000):
    """Particle swarm optimization on the sphere."""
    n_particles = 8
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    
    # Initialize particles
    particles = np.zeros((n_particles, N, 3))
    p_best = np.zeros((n_particles, N, 3))
    p_best_energy = np.zeros(n_particles)
    velocities = np.zeros((n_particles, N, 3))
    
    rng = np.random.RandomState(12345)
    
    for p in range(n_particles):
        if p == 0:
            particles[p] = points.copy()
        else:
            particles[p] = _project_to_sphere(points + rng.randn(N, 3) * 0.05)
        
        e = _log_energy(particles[p])
        p_best[p] = particles[p].copy()
        p_best_energy[p] = e
        
        if e < best_energy:
            best_energy = e
            best_points = particles[p].copy()
    
    # PSO parameters
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    w = 0.9   # inertia
    w_min = 0.4
    w_max = 0.9
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        # Update inertia
        w = w_min + (w_max - w_min) * (1.0 - it / max_iter)
        
        for p in range(n_particles):
            # Update velocity
            r1 = rng.rand(N, 3)
            r2 = rng.rand(N, 3)
            
            # Velocity in tangent space
            v_cog = c1 * r1 * (p_best[p] - particles[p])
            v_soc = c2 * r2 * (best_points - particles[p])
            
            new_vel = w * velocities[p] + v_cog + v_soc
            
            # Project velocity to tangent space
            dots = np.sum(new_vel * particles[p], axis=1, keepdims=True)
            new_vel = new_vel - dots * particles[p]
            
            # Limit velocity magnitude
            v_norms = np.linalg.norm(new_vel, axis=1, keepdims=True)
            v_max = 0.1
            mask_large = v_norms > v_max
            new_vel[mask_large] = new_vel[mask_large] * (v_max / v_norms[mask_large])
            
            velocities[p] = new_vel
            
            # Update position
            new_pos = particles[p] + new_vel
            new_pos = _project_to_sphere(new_pos)
            
            new_energy = _log_energy(new_pos)
            
            # Update personal best
            if new_energy < p_best_energy[p]:
                p_best_energy[p] = new_energy
                p_best[p] = new_pos.copy()
            
            # Update global best
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_pos.copy()
            
            particles[p] = new_pos
    
    return best_points, best_energy


def _gradient_descent_sphere(points, max_iter=10000, lr=0.01, start_time=None, deadline=None):
    """Gradient descent on the sphere with adaptive learning rate."""
    current_points = points.copy()
    best_points = points.copy()
    best_energy = _log_energy(current_points)
    current_energy = best_energy
    
    for i in range(max_iter):
        if start_time is not None and time.time() > deadline - 2.0:
            break
        
        n = len(current_points)
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current_points
        
        grads = weighted_sum - current_points * sum_weights[:, np.newaxis]
        
        dot_products = np.sum(grads * current_points, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current_points
        
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-10:
            break
        
        new_points = current_points - lr * grads_tangent
        new_points = _project_to_sphere(new_points)
        
        new_energy = _log_energy(new_points)
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
            lr = min(lr * 1.05, 0.1)
        else:
            lr = lr * 0.9
        
        if lr < 1e-10:
            break
    
    return best_points, best_energy


def _simulated_annealing(points, start_time, deadline, max_iter=50000):
    """Simulated annealing on the sphere."""
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    current_points = points.copy()
    current_energy = best_energy
    
    T = 0.01
    T_min = 1e-8
    cooling = 0.9995
    
    for _ in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        idx = np.random.randint(N)
        rand_vec = np.random.randn(3)
        tangent = rand_vec - np.dot(rand_vec, current_points[idx]) * current_points[idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-15:
            continue
        tangent = tangent / tangent_norm
        
        step_size = T * 2.0
        new_point = current_points[idx] + step_size * tangent
        new_point = new_point / np.linalg.norm(new_point)
        
        new_points = current_points.copy()
        new_points[idx] = new_point
        
        new_energy = _log_energy(new_points)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(T, T_min)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T = max(T * cooling, T_min)
    
    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
        _random_uniform_sphere(N, seed=6),
        _random_uniform_sphere(N, seed=7),
    ]
    
    rng = np.random.RandomState(42)
    for k in range(3):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.01
        inits.append(_project_to_sphere(perturbed))
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: L-BFGS-B optimization from multiple starts
    for idx, init in enumerate(inits):
        if time.time() > deadline - 50.0:
            break
        
        x0 = init.flatten()
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 2: PSO for global exploration
    if time.time() < deadline - 40.0:
        pso_points, pso_energy = _ps_optimize(
            best_points, start_time, deadline, max_iter=20000
        )
        if pso_energy < best_energy:
            best_energy = pso_energy
            best_points = pso_points.copy()
    
    # Phase 3: Gradient descent refinement
    if time.time() < deadline - 25.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, max_iter=15000, lr=0.005,
            start_time=start_time, deadline=deadline
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()
    
    # Phase 4: Simulated annealing for fine-tuning
    if time.time() < deadline - 15.0:
        sa_points, sa_energy = _simulated_annealing(
            best_points, start_time, deadline, max_iter=40000
        )
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_points = sa_points.copy()
    
    # Phase 5: Final high-precision L-BFGS-B polish
    if time.time() < deadline - 8.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 10000, 'maxfun': 100000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 6: Very tight final polish
    if time.time() < deadline - 3.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 200}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
