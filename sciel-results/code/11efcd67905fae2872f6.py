import numpy as np
from math import fsum
import time

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n, seed_shift=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5 + seed_shift
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_energy(x):
    """Compute logarithmic energy sum_{i<j} log(1/d_ij)."""
    n = len(x)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Only sum upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    log_terms = np.log(1.0 / dist) * mask
    return fsum(log_terms[mask].tolist())

def _gradient_descent(x, num_steps=2000, initial_lr=0.05, final_lr=0.001):
    """Run gradient descent on sphere to minimize log energy with adaptive step size."""
    x = x.copy()
    n = len(x)
    best_energy = _compute_energy(x)
    
    for step in range(num_steps):
        # Vectorized distance calculation
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        inv_dist_sq = 1.0 / dist_sq
        inv_dist_sq[~np.isfinite(inv_dist_sq)] = 0.0
        
        force = diff * inv_dist_sq[:, :, np.newaxis]
        total_force = np.sum(force, axis=1)
        
        # Project onto tangent plane
        dot_products = np.sum(total_force * x, axis=1)
        tangent_force = total_force - dot_products[:, np.newaxis] * x
        
        # Exponential decay of learning rate
        progress = step / num_steps
        lr = initial_lr * (final_lr / initial_lr) ** progress
        
        # Adaptive step size: try several multipliers and pick the one that reduces energy most
        best_step_x = x
        best_step_energy = best_energy
        for mult in [1.0, 0.5, 2.0, 0.25, 4.0]:
            trial_x = x + lr * mult * tangent_force
            # Normalize to unit sphere
            norms = np.linalg.norm(trial_x, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            trial_x = trial_x / norms
            
            # Quick energy check using vectorized computation
            diff_trial = trial_x[:, np.newaxis, :] - trial_x[np.newaxis, :, :]
            dist_trial = np.sqrt(np.sum(diff_trial ** 2, axis=2))
            np.fill_diagonal(dist_trial, np.inf)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            energy_trial = fsum((np.log(1.0 / dist_trial) * mask)[mask].tolist())
            
            if energy_trial < best_step_energy:
                best_step_energy = energy_trial
                best_step_x = trial_x
        
        x = best_step_x
        if best_step_energy < best_energy:
            best_energy = best_step_energy
    
    return x

def construct():
    # Try multiple initializations and keep the best
    best_x = None
    best_energy = np.inf
    
    # Initialization strategies
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci with various shifts
    for shift in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        initializations.append(_fibonacci_spiral(N, seed_shift=shift))
    
    # 3. Random initialization with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(4):
        rand = rng.randn(N, 3)
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        initializations.append(rand)
    
    # 4. Symmetric initialization: place points in rings
    ring_lats = np.linspace(-0.8, 0.8, 5)
    ring_points = []
    for lat in ring_lats:
        z = np.sin(lat)
        r = np.cos(lat)
        angles = np.linspace(0, 2*np.pi, 30, endpoint=False)
        ring = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(30, z)])
        ring_points.append(ring)
    initializations.append(np.vstack(ring_points))
    
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Run optimization with perturbation (Simulated Annealing style)
    for init in initializations:
        if not np.all(np.isfinite(init)):
            continue
        
        # Quick check: minimum distance
        diff = init[:, np.newaxis, :] - init[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if dist.min() < 1e-6:
            continue
        
        # Initial coarse optimization
        current_x = _gradient_descent(init, num_steps=500, initial_lr=0.05, final_lr=0.01)
        
        # Perturbation loop with more restarts and varying noise scales
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Varying noise scales: start larger, decrease over time
            if restart < 5:
                noise_scale = 0.1
            elif restart < 10:
                noise_scale = 0.05
            elif restart < 15:
                noise_scale = 0.02
            else:
                noise_scale = 0.01
            
            noise = rng.randn(N, 3) * noise_scale
            perturbed_x = current_x + noise
            # Normalize
            norms = np.linalg.norm(perturbed_x, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            perturbed_x = perturbed_x / norms
            
            # Fine optimization with more steps
            optimized = _gradient_descent(perturbed_x, num_steps=800, initial_lr=0.03, final_lr=0.0001)
            
            # Validate
            if not np.all(np.isfinite(optimized)):
                continue
            
            # Check minimum distance
            diff = optimized[:, np.newaxis, :] - optimized[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            if dist.min() < 1e-9:
                continue
            
            # Compute energy
            energy = _compute_energy(optimized)
            
            if energy < best_energy:
                best_energy = energy
                best_x = optimized
                current_x = optimized
        
        if time.time() - start_time > time_limit:
            break

    # Final fine-tuning pass on the best solution
    if best_x is not None and np.all(np.isfinite(best_x)):
        if time.time() - start_time < time_limit - 5.0:
            final_x = _gradient_descent(best_x, num_steps=1000, initial_lr=0.01, final_lr=0.00001)
            if np.all(np.isfinite(final_x)):
                diff = final_x[:, np.newaxis, :] - final_x[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist, np.inf)
                if dist.min() >= 1e-9:
                    final_energy = _compute_energy(final_x)
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_x = final_x
    
    # If all failed, fallback to Fibonacci
    if best_x is None or not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_spiral(N)
    
    return best_x

# EVOLVE-BLOCK-END
