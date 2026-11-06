"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _energy_contribution_fast(points, i):
    """Compute sum of log(1/d_ij) for point i with all j != i using NumPy."""
    diffs = points - points[i]
    dists = np.linalg.norm(diffs, axis=1)
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    dists = dists[mask]
    dists = np.maximum(dists, 1e-15)
    return np.sum(np.log(1.0 / dists))


def _move_and_eval_fast(points, i, step_size):
    """
    Move point i by a random step, return new point, delta energy.
    Uses fast NumPy-based energy calculation.
    """
    dir_vec = np.random.normal(0, 1, 3)
    dir_vec /= np.linalg.norm(dir_vec)
    
    new_point = points[i] + step_size * dir_vec
    new_point /= np.linalg.norm(new_point)
    
    old_contrib = _energy_contribution_fast(points, i)
    
    old_point = points[i]
    points[i] = new_point
    new_contrib = _energy_contribution_fast(points, i)
    points[i] = old_point
    
    delta_e = new_contrib - old_contrib
    return new_point, delta_e


def construct():
    """
    Use Simulated Annealing to minimize logarithmic energy.
    """
    best_points = None
    best_energy = float('inf')
    
    np.random.seed(42)
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(x0.copy())
    
    # 2. Random Uniform
    x1 = np.random.normal(0, 1, (N, 3))
    x1 /= np.linalg.norm(x1, axis=1, keepdims=True)
    configs.append(x1)
    
    # 3. Fibonacci with perturbation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + np.random.uniform(-0.2, 0.2, N)
    r = np.sqrt(1.0 - z * z)
    x2 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x2 += np.random.normal(0, 0.1, size=(N, 3))
    x2 /= np.linalg.norm(x2, axis=1, keepdims=True)
    configs.append(x2)

    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    for config in configs:
        if time.time() - start_time > time_limit:
            break
            
        points = config.copy()
        current_energy = _log_energy(points)
        
        # SA Parameters
        T = 0.1  # Initial temperature
        T_min = 1e-6
        alpha = 0.999  # Cooling rate
        steps_per_temp = 50  # Steps at each temperature
        step_size = 0.1
        
        # We will run SA until time limit or convergence
        while T > T_min:
            if time.time() - start_time > time_limit:
                break
                
            for _ in range(steps_per_temp):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random point
                i = np.random.randint(N)
                
                # Adaptive step size? Scale with T
                s = step_size * (1.0 + T)
                
                new_point, delta_e = _move_and_eval_fast(points, i, s)
                
                # Metropolis acceptance
                if delta_e < 0 or np.random.rand() < math.exp(-delta_e / T):
                    points[i] = new_point
                    current_energy += delta_e
            
            T *= alpha
        
        # Polish phase: Greedy local search
        # Try to improve each point with small steps
        polish_iterations = 500
        for _ in range(polish_iterations):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(N):
                if time.time() - start_time > time_limit:
                    break
                best_local = points[i].copy()
                best_local_e = _energy_contribution_fast(points, i)
                
                for _ in range(5):
                    dir_vec = np.random.normal(0, 1, 3)
                    dir_vec /= np.linalg.norm(dir_vec)
                    new_point = points[i] + 0.01 * dir_vec
                    new_point /= np.linalg.norm(new_point)
                    
                    old_point = points[i]
                    points[i] = new_point
                    new_e = _energy_contribution_fast(points, i)
                    points[i] = old_point
                    
                    if new_e < best_local_e:
                        best_local_e = new_e
                        best_local = new_point
                        
                if not np.allclose(best_local, points[i]):
                    points[i] = best_local
                    improved = True
                    # Recalculate full energy occasionally to avoid drift?
                    # Or just trust the incremental updates.
                    # Let's recalculate every 10 points to be safe
                    if i % 10 == 0:
                        current_energy = _log_energy(points)
                        
            if not improved:
                break
        
        current_energy = _log_energy(points)
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()

    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
# EVOLVE-BLOCK-END
