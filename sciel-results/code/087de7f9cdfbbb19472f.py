import numpy as np
import time
import math

N = 150

def _log_energy(x):
    """
    Compute the logarithmic energy efficiently.
    x: (N, 3) array of points on the unit sphere.
    """
    # Compute all pairwise distances
    # diff[i, j] = x[i] - x[j]
    # dist[i, j] = ||x[i] - x[j]||
    
    # To save memory and time for N=150, we can do it in chunks or just use full matrix
    # 150x150 is small enough for full matrix
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) and diagonal
    # dist = sqrt(dist_sq)
    # We need sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
    
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # safe_dist: replace 0 with 1 to avoid -inf, but we handle mask separately
    # Actually, we only sum over mask.
    dist = np.sqrt(dist_sq)
    
    # Clip distances to avoid log(0) if points are too close, though we check validity later
    # For energy calculation, if dist < 1e-15, energy is +inf (bad)
    # log(1/d) = -log(d)
    
    # Use math.fsum for precision as per task description? 
    # "pair terms summed with math.fsum"
    # The evaluator uses math.fsum. We should try to match precision or at least be consistent.
    # However, numpy sum is usually sufficient for comparison if we are not on the edge.
    # But to be safe, let's use a stable sum.
    
    valid_dists = dist[mask]
    
    # If any distance is 0, energy is inf
    if np.any(valid_dists < 1e-12):
        return float('inf')
        
    log_terms = -np.log(valid_dists)
    
    # The task says: "summed with math.fsum"
    # Converting to list for math.fsum might be slow for 150*149/2 ~ 11k terms.
    # math.fsum is O(N) but with high constant.
    # Let's check if numpy sum is close enough. 
    # Actually, for the search, we just need a consistent metric. 
    # For the final return, the evaluator computes it.
    # So we just need to minimize the true value.
    # Using np.sum is fine for the search heuristic.
    
    return float(np.sum(log_terms))

def _energy_and_grad(x):
    """
    Compute energy and Euclidean gradient.
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient of log(1/||x_i - x_j||) w.r.t x_i
    # d/dx_i log(1/||x_i - x_j||) = - d/dx_i log(||x_i - x_j||)
    # = - (1/||x_i - x_j||) * (x_i - x_j) / ||x_i - x_j||
    # = - (x_i - x_j) / ||x_i - x_j||^2
    
    # Sum over j != i
    # grad_i = sum_{j != i} - (x_i - x_j) / dist_ij^2
    
    # Create inverse distance squared matrix
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    # weighted_diff[i, j] = (x_i - x_j) * inv_dist_sq[i, j]
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    # grad_i = - sum_j weighted_diff[i, j]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _project_to_sphere(x):
    """
    Project points to unit sphere.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def _fibonacci_sphere(n):
    """
    Generate n points on the unit sphere using Fibonacci lattice.
    This provides excellent global dispersion.
    """
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        # z coordinate
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points[i] = [x, y, z]
    return points

def _local_optimize(x, time_limit=5.0, tol=1e-10):
    """
    Local optimization using Riemannian Gradient Descent with backtracking.
    Fast and effective for polishing.
    """
    start_time = time.time()
    energy, grad_eucl = _energy_and_grad(x)
    best_energy = energy
    best_x = x.copy()
    
    step_size = 0.01
    
    while time.time() - start_time < time_limit:
        # Project gradient to tangent space
        # grad_tan = grad_eucl - (grad_eucl . x) * x
        proj = np.sum(grad_eucl * x, axis=1, keepdims=True)
        grad_tan = grad_eucl - proj * x
        
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
            
        # Normalize direction
        direction = grad_tan / (grad_norm + 1e-12)
        
        # Line search: try a few step sizes
        found_step = False
        for alpha in [step_size, step_size*0.5, step_size*0.25, step_size*0.1]:
            # Move along geodesic (approx by exp map or simple projection for small steps)
            # Simple projection is faster for small steps
            x_new = x - alpha * grad_eucl
            x_new = _project_to_sphere(x_new)
            
            e_new, _ = _energy_and_grad(x_new)
            
            if e_new < energy:
                x = x_new
                energy = e_new
                step_size = min(alpha * 1.5, 0.1) # Increase step if successful
                found_step = True
                if e_new < best_energy:
                    best_energy = e_new
                    best_x = x.copy()
                break
        if not found_step:
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return best_x, best_energy

def _simulated_annealing(x_init, time_limit=10.0):
    """
    Simulated Annealing on the sphere.
    Moves: Pick a random pair, rotate one point slightly around the axis defined by their cross product?
    Or simpler: Pick a random point, move it in a random tangent direction.
    
    Actually, for log energy, large moves are often needed to escape deep wells of close pairs.
    Let's use a "kick" strategy: pick a random point, move it by a random amount in a random direction.
    """
    start_time = time.time()
    x = x_init.copy()
    energy, _ = _energy_and_grad(x)
    best_energy = energy
    best_x = x.copy()
    
    T = 0.1 # Initial temperature
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < 1e-6:
            break
            
        # Pick a random index
        idx = np.random.randint(N)
        
        # Generate random tangent vector at x[idx]
        # Random vector v, project out x[idx]
        v = np.random.normal(size=3)
        v -= np.dot(v, x[idx]) * x[idx]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:
            continue
        v /= v_norm
        
        # Step size proportional to temperature
        step = T * 0.5
        x_new = x.copy()
        x_new[idx] = x_new[idx] + step * v
        x_new[idx] = _project_to_sphere(x_new[idx])
        
        e_new, _ = _energy_and_grad(x_new)
        
        # Accept if better, or with probability exp(-(e_new - energy)/T)
        if e_new < energy:
            x = x_new
            energy = e_new
            if e_new < best_energy:
                best_energy = e_new
                best_x = x.copy()
        else:
            delta_e = e_new - energy
            if delta_e < 100 * T: # Avoid overflow
                prob = np.exp(-delta_e / T)
                if np.random.random() < prob:
                    x = x_new
                    energy = e_new
                    
    return best_x, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Initialization strategies
    # 1. Fibonacci sphere (very good uniform distribution)
    # 2. Random points
    # 3. Halton-like (low discrepancy)
    
    inits = []
    
    # Fibonacci
    inits.append(_fibonacci_sphere(N))
    
    # Random
    rng = np.random.default_rng(42)
    x_rand = rng.normal(size=(N, 3))
    x_rand = _project_to_sphere(x_rand)
    inits.append(x_rand)
    
    # Slightly perturbed Fibonacci
    x_fib_pert = _fibonacci_sphere(N)
    x_fib_pert += 0.1 * rng.normal(size=(N, 3))
    x_fib_pert = _project_to_sphere(x_fib_pert)
    inits.append(x_fib_pert)
    
    # Another random
    x_rand2 = rng.normal(size=(N, 3))
    x_rand2 = _project_to_sphere(x_rand2)
    inits.append(x_rand2)
    
    # Run optimization for each init
    # Strategy: 
    # 1. Quick local polish (RCG/PGD)
    # 2. If time remains, do Simulated Annealing from the polished state or a new random state.
    
    for i, x_init in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # Phase 1: Local Optimization
        # Use a short time limit for local opt to ensure we try multiple starts
        local_time = 10.0
        x_opt, e_opt = _local_optimize(x_init, time_limit=local_time)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
            
        # Phase 2: Simulated Annealing to escape local minima
        # Only if we have enough time left
        if time.time() < deadline - 15:
            sa_time = min(15.0, deadline - 15 - time.time())
            if sa_time > 2.0:
                # Run SA from the local minimum
                x_sa, e_sa = _simulated_annealing(x_opt, time_limit=sa_time)
                
                if e_sa < best_energy:
                    best_energy = e_sa
                    best_x = x_sa.copy()
                    
                # Polish the SA result
                if time.time() < deadline - 5:
                    polish_time = min(5.0, deadline - 5 - time.time())
                    if polish_time > 1.0:
                        x_final, e_final = _local_optimize(x_sa, time_limit=polish_time)
                        if e_final < best_energy:
                            best_energy = e_final
                            best_x = x_final.copy()
                            
    # Final check: if best_x is None, use the first init
    if best_x is None:
        best_x = inits[0]
        
    # Final validation and cleanup
    # Ensure points are on sphere
    best_x = _project_to_sphere(best_x)
    
    return best_x

# EVOLVE-BLOCK-END
