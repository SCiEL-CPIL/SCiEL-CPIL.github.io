"""
Strategy: Global stochastic search via Simulated Annealing on the unit sphere.
Unlike local gradient methods, SA accepts uphill moves to escape local minima.
We use a multi-restart approach with a strict time budget to maximize the
chance of finding a deeper global minimum within the 150s limit.
"""
import numpy as np
import time
import math

N = 150
DEADLINE = 145.0  # Safety margin for the 150s limit

def _energy(points):
    """
    Compute logarithmic energy: sum_{i<j} log(1 / |x_i - x_j|)
    Vectorized for speed.
    """
    # Compute all pairwise distances
    # points is (N, 3)
    # dist_sq[i, j] = |x_i - x_j|^2
    # We want sum over i < j of log(1/sqrt(dist_sq)) = -0.5 * sum_{i<j} log(dist_sq)
    
    # Efficient way: compute squared distances
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    # Since on unit sphere, ||x_i||^2 = 1
    # So dist_sq = 2 - 2 * (X @ X.T)
    
    G = points @ points.T
    dist_sq = 2.0 - 2.0 * G
    # Ensure positive for log (numerical safety)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    # We need sum over i < j
    # Create mask for upper triangle
    iu = np.triu_indices(N, k=1)
    vals = dist_sq[iu]
    
    # Energy = sum log(1/sqrt(d)) = sum (-0.5 log(d^2)) = -0.5 sum log(dist_sq)
    # Using math.fsum as per contract for precision, but numpy sum is usually fine for comparison.
    # However, contract says "pair terms summed with math.fsum". 
    # For performance in SA, we might use np.sum. But for the final return, we should be precise.
    # Let's use np.sum for the optimization loop speed, and rely on the evaluator's fsum.
    # Actually, let's just compute the value. The evaluator computes the score.
    # We just need to return the points. The score is computed by the evaluator.
    # So _energy is only used for comparison in the algorithm.
    
    return -0.5 * np.sum(np.log(vals))

def _random_sphere(n):
    """Generate n random points on unit sphere."""
    # Normal distribution trick
    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid zero norms (extremely unlikely)
    norms[norms == 0] = 1.0
    return v / norms

def _simulated_annealing(points, t_start=1.0, t_end=0.01, cooling=0.9995, max_iter=5000, time_limit=10.0):
    """
    Run SA on the given points.
    Returns best points found.
    """
    current = points.copy()
    current_energy = _energy(current)
    best = current.copy()
    best_energy = current_energy
    
    start_time = time.time()
    t = t_start
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb: pick random points and move them slightly
        # Move along random tangent directions
        n_moves = 3
        idxs = np.random.choice(N, n_moves, replace=False)
        
        new_current = current.copy()
        for idx in idxs:
            # Random tangent vector
            rand_vec = np.random.normal(size=3)
            # Project onto tangent plane: v - (v.x) x
            dot = np.dot(rand_vec, current[idx])
            tangent = rand_vec - dot * current[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t < 1e-10:
                continue
            tangent /= norm_t
            
            # Step size proportional to temperature
            step_size = 0.1 * t
            new_point = current[idx] + step_size * tangent
            # Renormalize
            new_point /= np.linalg.norm(new_point)
            new_current[idx] = new_point
        
        new_energy = _energy(new_current)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.rand() < math.exp(-delta / t):
            current = new_current
            current_energy = new_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
        t *= cooling
        if t < t_end:
            break
            
    return best, best_energy

def construct():
    start_time = time.time()
    
    # Initial best
    # Start with Fibonacci spiral as a good initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = fib_points
    best_energy = _energy(fib_points)
    
    # Multi-restart SA
    restarts = 0
    while time.time() < DEADLINE:
        # Check time
        if time.time() > DEADLINE - 10.0:
            break
            
        # Generate a random starting point
        # Mix: sometimes random, sometimes perturbed Fibonacci
        if np.random.rand() < 0.3:
            start_points = _random_sphere(N)
        else:
            start_points = fib_points + 0.05 * np.random.normal(size=(N, 3))
            start_points /= np.linalg.norm(start_points, axis=1, keepdims=True)
            
        # Run SA
        # Adjust time limit for SA based on remaining time
        remaining = DEADLINE - time.time()
        sa_time_limit = min(remaining * 0.9, 20.0) # Cap SA time per restart to allow many restarts
        
        if sa_time_limit < 1.0:
            break
            
        sa_points, sa_energy = _simulated_annealing(
            start_points, 
            t_start=0.5, 
            t_end=0.001, 
            cooling=0.999, 
            max_iter=10000, 
            time_limit=sa_time_limit
        )
        
        if sa_energy < best_energy:
            best_points = sa_points
            best_energy = sa_energy
            
        restarts += 1
        
    return best_points
