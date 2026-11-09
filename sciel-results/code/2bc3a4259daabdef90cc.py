"""Packing 50 equal circles in a unit circle using PSO + local refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """Compute the minimum radius for a given configuration.
    
    Returns the largest common radius r such that all circles fit.
    """
    positions = np.asarray(positions)
    n = positions.shape[0]
    
    # Boundary constraint: min_i (1 - |c_i|)
    dists_to_origin = np.linalg.norm(positions, axis=1)
    boundary_min = 1.0 - np.max(dists_to_origin)
    
    # Pairwise constraint: min_{i<j} |c_i - c_j| / 2
    # Compute all pairwise distances efficiently
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    pairwise_min = np.min(dists) / 2.0
    
    return min(boundary_min, pairwise_min)


def _penalized_score(positions, penalty_factor=10.0):
    """Penalized score that discourages violations."""
    positions = np.asarray(positions)
    n = positions.shape[0]
    
    # Boundary constraint
    dists_to_origin = np.linalg.norm(positions, axis=1)
    boundary_violations = np.maximum(0, dists_to_origin - 1.0)
    boundary_min = 1.0 - np.max(dists_to_origin)
    
    # Pairwise constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    pairwise_min = np.min(dists) / 2.0
    
    r = min(boundary_min, pairwise_min)
    
    # Penalty for violations
    penalty = penalty_factor * (np.sum(boundary_violations) + 
                                 np.sum(np.maximum(0, -pairwise_min)))
    
    return r - penalty


def _hex_lattice_init(n_circles, seed=42):
    """Generate initial positions using a hexagonal lattice pattern."""
    rng = np.random.RandomState(seed)
    
    # Try different spacings
    for spacing in [0.22, 0.21, 0.20, 0.19, 0.18]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= n_circles:
            # Select the n_circles closest to center
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')[:n_circles]
            return points[idx].copy()
    
    # Fallback: random points
    return rng.uniform(-0.8, 0.8, size=(n_circles, 2))


def _local_refine(positions, max_iters=100, tol=1e-8):
    """Refine a configuration using L-BFGS-B to maximize the minimum radius."""
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    
    def neg_score(x):
        pos = x.reshape(n, 2)
        return -_score(pos)
    
    # Use L-BFGS-B with bounds to keep points inside the unit circle
    bounds = [(-0.99, 0.99)] * (2 * n)
    
    result = minimize(neg_score, positions.flatten(), method='L-BFGS-B', 
                      bounds=bounds, options={'maxiter': max_iters, 'ftol': tol})
    
    if result.success or result.fun < neg_score(positions.flatten()):
        return result.x.reshape(n, 2)
    return positions


def _pso_optimize(n_particles=20, max_iter=50, w=0.7, c1=1.5, c2=1.5,
                  init_positions=None, time_limit=100.0):
    """Particle Swarm Optimization for circle packing."""
    start_time = time.time()
    
    # Initialize particles
    if init_positions is not None:
        best_init = init_positions.copy()
        best_score = _score(best_init)
    else:
        best_init = _hex_lattice_init(N)
        best_score = _score(best_init)
    
    # Create initial swarm
    particles = np.zeros((n_particles, N, 2))
    p_scores = np.zeros(n_particles)
    
    for i in range(n_particles):
        if i == 0:
            particles[i] = best_init
        else:
            # Perturb from best init
            particles[i] = best_init + np.random.randn(N, 2) * 0.05
            # Clip to unit circle
            norms = np.linalg.norm(particles[i], axis=1)
            mask = norms > 0.98
            particles[i][mask] *= 0.98 / norms[mask]
        
        p_scores[i] = _score(particles[i])
    
    # Track bests
    p_best = particles.copy()
    p_best_scores = p_scores.copy()
    g_best = best_init.copy()
    g_best_score = best_score
    
    # Update global best
    for i in range(n_particles):
        if p_scores[i] > g_best_score:
            g_best_score = p_scores[i]
            g_best = particles[i].copy()
    
    velocities = np.zeros_like(particles)
    
    for it in range(max_iter):
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        for i in range(n_particles):
            # Update velocity
            r1 = np.random.rand(N, 2)
            r2 = np.random.rand(N, 2)
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (p_best[i] - particles[i]) + 
                           c2 * r2 * (g_best - particles[i]))
            
            # Update position
            particles[i] += velocities[i]
            
            # Keep inside unit circle
            norms = np.linalg.norm(particles[i], axis=1)
            mask = norms > 0.99
            if np.any(mask):
                particles[i][mask] *= 0.99 / norms[mask]
            
            # Evaluate
            p_scores[i] = _score(particles[i])
            
            # Update particle best
            if p_scores[i] > p_best_scores[i]:
                p_best_scores[i] = p_scores[i]
                p_best[i] = particles[i].copy()
            
            # Update global best
            if p_scores[i] > g_best_score:
                g_best_score = p_scores[i]
                g_best = particles[i].copy()
        
        # Adaptive inertia weight
        w = 0.7 - 0.3 * (it / max_iter)
    
    return g_best, g_best_score


def construct():
    """Construct the optimal packing of 50 circles in a unit circle."""
    total_start = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Phase 1: Generate good initial solution
    init = _hex_lattice_init(N, seed=42)
    
    # Phase 2: Local refinement of initial solution
    refined = _local_refine(init, max_iters=50, tol=1e-7)
    best_pos = refined.copy()
    best_score = _score(refined)
    
    # Phase 3: PSO to explore multiple local optima
    if time.time() - total_start < time_limit - 30:
        pso_result, pso_score = _pso_optimize(
            n_particles=15, 
            max_iter=30, 
            init_positions=refined,
            time_limit=time_limit - (time.time() - total_start) - 10
        )
        if pso_score > best_score:
            best_pos = pso_result
            best_score = pso_score
    
    # Phase 4: Final local refinement
    if time.time() - total_start < time_limit - 15:
        final = _local_refine(best_pos, max_iters=200, tol=1e-9)
        final_score = _score(final)
        if final_score > best_score:
            best_pos = final
            best_score = final_score
    
    # Phase 5: One more refinement with tighter tolerance
    if time.time() - total_start < time_limit - 5:
        final2 = _local_refine(best_pos, max_iters=300, tol=1e-10)
        final2_score = _score(final2)
        if final2_score > best_score:
            best_pos = final2
            best_score = final2_score
    
    return best_pos
# EVOLVE-BLOCK-END
