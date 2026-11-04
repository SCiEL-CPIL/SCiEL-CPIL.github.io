"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _radius(c):
    """Compute the packing radius for a configuration c of shape (N, 3)."""
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    dists = pdist(c, metric='euclidean')
    if len(dists) > 0:
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
    return min(min_wall, min_pair / 2.0)

def _objective(c_flat):
    """Objective function to minimize (negative radius)."""
    c = c_flat.reshape(N, 3)
    c = np.clip(c, -0.5, 0.5)
    r = _radius(c)
    return -r

def _local_refine(c_flat, max_iter=20, ftol=1e-8):
    """
    Perform local optimization step using L-BFGS-B.
    """
    try:
        bounds = [(-0.5, 0.5)] * (N * 3)
        result = minimize(_objective, c_flat, method='L-BFGS-B', 
                          bounds=bounds, 
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-5})
        if result.fun < _objective(c_flat):
            return result.x
        else:
            return c_flat
    except Exception:
        return c_flat

def _generate_lattice_init():
    """
    Generate an initial configuration based on a simple cubic lattice.
    """
    # Use a 5x5x4 grid (100 points)
    nx, ny, nz = 5, 5, 4
    points = []
    a = 1.0 / max(nx, ny, nz) * 0.98
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i - (nx - 1) / 2.0) * a
                y = (j - (ny - 1) / 2.0) * a
                z = (k - (nz - 1) / 2.0) * a
                points.append([x, y, z])
                
    points = np.array(points[:N])
    points += np.random.randn(N, 3) * 0.005
    return points.ravel()

def construct():
    np.random.seed(42)
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    swarm_size = 15
    w_start = 0.9
    w_end = 0.4
    c1 = 1.8
    c2 = 1.2
    
    # Initialize swarm
    particles = np.zeros((swarm_size, N * 3))
    
    # Base lattice
    base_lattice = _generate_lattice_init()
    
    for i in range(swarm_size):
        if i == 0:
            particles[i] = base_lattice
        elif i < 5:
            # Perturbed lattice with varying noise
            noise_level = 0.02 * (i + 1)
            particles[i] = base_lattice + np.random.randn(N * 3) * noise_level
        elif i < 10:
            # Different lattice orientation or spacing
            scale = 0.8 + 0.04 * (i - 4)
            pts = _generate_lattice_init()
            pts = pts.reshape(N, 3) * scale
            particles[i] = pts.ravel()
        else:
            # Random
            particles[i] = np.random.uniform(-0.4, 0.4, N * 3)
            
    # Evaluate initial positions and apply local refinement
    personal_bests = np.zeros_like(particles)
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        if time.time() - start_time > time_limit:
            break
        refined = _local_refine(particles[i], max_iter=15, ftol=1e-7)
        particles[i] = refined
        val = _objective(particles[i])
        personal_best_vals[i] = val
        personal_bests[i] = particles[i].copy()
        
    global_best_idx = np.argmin(personal_best_vals)
    global_best = personal_bests[global_best_idx].copy()
    global_best_val = personal_best_vals[global_best_idx]
    
    velocities = np.zeros_like(particles)
    
    # Adaptive PSO loop
    max_iter = 50
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Adaptive inertia
        w = w_start - (w_start - w_end) * (it / max_iter)
        
        for i in range(swarm_size):
            if time.time() - start_time > time_limit:
                break
                
            r1 = np.random.rand(N * 3)
            r2 = np.random.rand(N * 3)
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_bests[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            # Velocity clamping to prevent explosion
            v_max = 0.15
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            particles[i] += velocities[i]
            
            # Clip to bounds
            particles[i] = np.clip(particles[i], -0.49, 0.49)
            
            # Local refinement step
            # Reduce iterations as we get closer to the end
            iter_limit = 10 if it < max_iter // 2 else 5
            refined = _local_refine(particles[i], max_iter=iter_limit, ftol=1e-7)
            particles[i] = refined
            
            # Evaluate
            val = _objective(particles[i])
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                
                if val < global_best_val:
                    global_best_val = val
                    global_best = particles[i].copy()
                    
    # Final refinement on the best solution
    if time.time() - start_time < time_limit - 10:
        result = minimize(_objective, global_best, method='L-BFGS-B', 
                          bounds=[(-0.5, 0.5)] * (N * 3), 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
        global_best = result.x
    
    c_final = np.clip(global_best.reshape(N, 3), -0.5, 0.5)
    
    return c_final
# EVOLVE-BLOCK-END
