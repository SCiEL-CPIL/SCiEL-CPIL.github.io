"""PSO-based circle packing with log-sum-exp surrogate and coordinate polish."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    boundary_min = 1.0 - np.max(norms)
    if N > 1:
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)


def _surrogate_fitness(points, beta=50.0):
    """
    Smooth surrogate: minimize log-sum-exp of violations.
    Lower is better. Uses pairwise distances and boundary distances.
    """
    norms = np.hypot(points[:, 0], points[:, 1])
    # Boundary violation: want norms <= 1 - r, but we use a soft target
    # Actually, for PSO we maximize a fitness that approximates the min radius.
    # Let's define fitness = - (logsumexp(pairwise_violations) + logsumexp(boundary_violations))
    # where violations are based on a target radius r_target.
    # But we want to find the max r.
    # Alternative: use a smooth approximation of the min distance.
    # min(d_ij) ~ -logsumexp(-beta * d_ij) / beta
    # min(1 - |c_i|) ~ -logsumexp(-beta * (1 - |c_i|)) / beta
    
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    
    # Smooth min of pairwise distances
    # To avoid overflow, shift by max
    d_flat = d[d < np.inf]
    if len(d_flat) == 0:
        return 0.0
    max_d = np.max(d_flat)
    # logsumexp(-beta * (d - max_d)) + beta * max_d
    log_sum = np.log(np.sum(np.exp(-beta * (d_flat - max_d))))
    smooth_min_pair = -(log_sum + beta * max_d) / beta  # This is actually -logsumexp(-beta*d)/beta
    # Wait, min(x) = -1/beta * log(sum(exp(-beta*x)))
    # So smooth_min_pair = -1/beta * log(sum(exp(-beta * d)))
    # Using shift: log(sum(exp(-beta * (d - max_d)))) = log(sum(exp(-beta*d) * exp(beta*max_d))) = log(sum(exp(-beta*d))) + beta*max_d
    # So log(sum(exp(-beta*d))) = log_sum - beta*max_d
    # smooth_min_pair = -(log_sum - beta*max_d)/beta = max_d - log_sum/beta
    
    # Boundary distances
    b_dist = 1.0 - norms
    max_b = np.max(b_dist)
    log_sum_b = np.log(np.sum(np.exp(-beta * (b_dist - max_b))))
    smooth_min_boundary = max_b - log_sum_b / beta
    
    # The actual score is min(smooth_min_pair, smooth_min_boundary)
    # For fitness, we want to maximize this.
    # But PSO typically minimizes. Let's return negative.
    return -min(smooth_min_pair, smooth_min_boundary)


def _pso_optimize(initial_points, n_particles=10, max_iters=200, w=0.9, c1=1.5, c2=1.5):
    """
    Particle Swarm Optimization.
    Each particle is an array of shape (N, 2).
    """
    global START_TIME
    
    # Initialize swarm
    # Particle 0 is the initial points
    positions = np.zeros((n_particles, N, 2))
    velocities = np.zeros((n_particles, N, 2))
    
    # Seed particle 0
    positions[0] = initial_points
    # Seed others with random perturbations
    for i in range(1, n_particles):
        positions[i] = initial_points + np.random.normal(0, 0.05, (N, 2))
        # Ensure inside unit disk
        norms = np.hypot(positions[i][:, 0], positions[i][:, 1])
        mask = norms > 0.99
        if np.any(mask):
            positions[i][mask] = positions[i][mask] / norms[mask, np.newaxis] * 0.99
            
    # Evaluate initial fitness
    fitnesses = np.array([_surrogate_fitness(pos) for pos in positions])
    
    # Personal bests
    pbest_pos = positions.copy()
    pbest_fit = fitnesses.copy()
    
    # Global best
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Update velocities and positions
        r1 = np.random.rand(n_particles, 1, 1)
        r2 = np.random.rand(n_particles, 1, 1)
        
        velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        
        # Limit velocity to prevent divergence
        max_vel = 0.1
        vel_norms = np.linalg.norm(velocities, axis=2, keepdims=True)
        mask = vel_norms > max_vel
        if np.any(mask):
            velocities[mask] = velocities[mask] / vel_norms[mask] * max_vel
            
        positions += velocities
        
        # Enforce boundary constraint: project back to unit disk if outside
        norms = np.hypot(positions[:, :, 0], positions[:, :, 1])
        mask = norms > 0.99
        if np.any(mask):
            # Flatten for easier masking
            flat_pos = positions.reshape(-1, 2)
            flat_norms = norms.reshape(-1)
            flat_mask = flat_norms > 0.99
            flat_pos[flat_mask] = flat_pos[flat_mask] / flat_norms[flat_mask, np.newaxis] * 0.99
            positions = flat_pos.reshape(n_particles, N, 2)
            
        # Evaluate new fitnesses
        new_fitnesses = np.array([_surrogate_fitness(pos) for pos in positions])
        
        # Update personal bests
        improved = new_fitnesses < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = new_fitnesses[improved]
        
        # Update global best
        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()
            
        # Decay inertia weight
        w = max(0.4, w * 0.99)
        
    return gbest_pos


def _coordinate_polish(points, max_iters=1000):
    """
    Coordinate descent polish: move each point one at a time to maximize min distance.
    """
    global START_TIME
    points = points.copy()
    current_score = _score(points)
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 2:
            break
            
        improved = False
        for i in range(N):
            # Try to move point i to maximize the minimum distance
            # Current point
            p = points[i]
            
            # Calculate distances to all other points and boundary
            d_to_others = np.linalg.norm(points - p, axis=1)
            np.fill_diagonal(d_to_others, np.inf)
            min_d_others = np.min(d_to_others)
            d_to_boundary = 1.0 - np.linalg.norm(p)
            
            current_min = min(min_d_others, d_to_boundary)
            
            # Try small steps in 4 directions
            best_p = p
            best_min = current_min
            step = 0.001
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                           (step/np.sqrt(2), step/np.sqrt(2)), (-step/np.sqrt(2), step/np.sqrt(2)),
                           (step/np.sqrt(2), -step/np.sqrt(2)), (-step/np.sqrt(2), -step/np.sqrt(2))]:
                new_p = p + np.array([dx, dy])
                new_norm = np.linalg.norm(new_p)
                if new_norm > 0.999:
                    continue
                    
                new_d_to_others = np.linalg.norm(points - new_p, axis=1)
                np.fill_diagonal(new_d_to_others, np.inf)
                new_min_d_others = np.min(new_d_to_others)
                new_d_to_boundary = 1.0 - new_norm
                new_min = min(new_min_d_others, new_d_to_boundary)
                
                if new_min > best_min + 1e-8:
                    best_min = new_min
                    best_p = new_p
                    
            if not np.array_equal(best_p, p):
                points[i] = best_p
                improved = True
                
        if not improved:
            break
            
    final_score = _score(points)
    return points, final_score


def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]


def _ring_initial():
    """Generate ring-based initial configuration."""
    points = []
    points.append([0.0, 0.0])
    radii = [0.3, 0.55, 0.75, 0.9]
    counts = [6, 12, 18, 13]
    for r, n in zip(radii, counts):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    initializations = []
    
    # Hexagonal
    for spacing in [0.22, 0.24, 0.26]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
            
    # Ring
    initializations.append(_ring_initial())
    
    # Random
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.uniform(-0.9, 0.9, (N, 2))
        norms = np.hypot(pts[:, 0], pts[:, 1])
        mask = norms > 0.9
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.9
        initializations.append(pts)
        
    for init_points in initializations:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        # Run PSO
        pso_result = _pso_optimize(init_points, n_particles=8, max_iters=100)
        
        # Polish
        polished, score = _coordinate_polish(pso_result)
        
        if score > best_score:
            best_score = score
            best_points = polished.copy()
            
    # Final polish on best
    if best_points is not None:
        polished, score = _coordinate_polish(best_points, max_iters=2000)
        if score > best_score:
            best_score = score
            best_points = polished
            
    # Fallback
    if best_points is None:
        pts = _hexagonal_initial(0.25)
        if pts is not None:
            best_points = pts
        else:
            best_points = np.zeros((N, 2))
            
    # Ensure validity
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999
        
    return best_points
