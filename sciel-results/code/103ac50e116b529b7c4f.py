"""Packing 50 equal circles in a unit circle using PSO with local polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 120s limit


def _score(positions):
    """Compute the common radius for a given set of centers."""
    # Distance from origin
    dists_origin = np.sqrt(np.sum(positions ** 2, axis=1))
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(pair_dists, np.inf)
    min_pair = np.min(pair_dists)
    r = min(min_pair / 2.0, 1.0 - np.max(dists_origin))
    return r


def _pso_score(positions):
    """PSO fitness: maximize r, penalize invalid configurations."""
    dists_origin = np.sqrt(np.sum(positions ** 2, axis=1))
    if np.any(dists_origin >= 1.0):
        # At least one circle is outside or on boundary
        overflow = np.max(dists_origin) - 1.0
        return -1.0 - 10.0 * overflow
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(pair_dists, np.inf)
    min_pair = np.min(pair_dists)
    r = min(min_pair / 2.0, 1.0 - np.max(dists_origin))
    return r


def _local_polish(positions, max_iters=50):
    """Local optimization using L-BFGS-B to maximize the minimum distance."""
    def objective(x):
        pos = x.reshape(N, 2)
        dists_origin = np.sqrt(np.sum(pos ** 2, axis=1))
        if np.any(dists_origin >= 1.0):
            return 1.0 + np.max(dists_origin)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(pair_dists, np.inf)
        min_pair = np.min(pair_dists)
        return -min(min_pair / 2.0, 1.0 - np.max(dists_origin))
    
    x0 = positions.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iters, 'ftol': 1e-10})
    return result.x.reshape(N, 2)


def _hex_init():
    """Generate initial hexagonal lattice configuration."""
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


def construct():
    start_time = time.time()
    
    # Initial configuration
    best_pos = _hex_init()
    best_score = _score(best_pos)
    
    # PSO setup
    n_particles = 10
    n_dims = 2 * N
    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive coefficient
    c2 = 1.5  # social coefficient
    
    # Initialize swarm
    particles = np.zeros((n_particles, n_dims))
    velocities = np.zeros((n_particles, n_dims))
    pbest_positions = np.zeros((n_particles, n_dims))
    pbest_scores = np.zeros(n_particles)
    gbest_position = best_pos.flatten()
    gbest_score = best_score
    
    for i in range(n_particles):
        # Random initialization within unit circle
        while True:
            pos = np.random.uniform(-1, 1, size=(N, 2))
            if np.all(np.sqrt(np.sum(pos ** 2, axis=1)) < 1.0):
                break
        particles[i] = pos.flatten()
        score = _pso_score(pos)
        pbest_positions[i] = pos.flatten()
        pbest_scores[i] = score
        if score > gbest_score:
            gbest_score = score
            gbest_position = pos.flatten()
    
    # PSO iteration
    max_iterations = 200
    for it in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        for i in range(n_particles):
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (pbest_positions[i] - particles[i]) + 
                           c2 * r2 * (gbest_position - particles[i]))
            
            particles[i] += velocities[i]
            
            # Clip to valid range
            particles[i] = np.clip(particles[i], -1.0, 1.0)
            
            pos = particles[i].reshape(N, 2)
            score = _pso_score(pos)
            
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = particles[i].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = particles[i].copy()
        
        # Check deadline
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
    
    # Final local polish on the best configuration found
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        best_pos = gbest_position.reshape(N, 2)
        polished = _local_polish(best_pos, max_iters=100)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_pos = polished
            best_score = polished_score
    
    return best_pos
