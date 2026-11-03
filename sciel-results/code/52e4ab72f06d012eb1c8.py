import numpy as np
import time

N = 100

def _energy_and_grad(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad_tan = grad - dot * points
    
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    energy = -0.25 * np.sum(log_dist_sq)
    
    return energy, grad_tan

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_energy, _ = _energy_and_grad(points)
    best_points = points.copy()
    current_energy = best_energy
    current_points = points.copy()
    
    def step_descent(pts, energy, steps, lr_start, lr_end):
        nonlocal best_energy, best_points
        p = pts.copy()
        e = energy
        for _ in range(steps):
            if time.time() > deadline:
                return p, e, True
            
            # Linear interpolation of lr? Or geometric?
            # Geometric is better for decay.
            # lr = lr_start * (lr_end/lr_start) ** (step/steps)
            # Since we don't have step index easily in this loop structure without passing it, 
            # let's just use a constant lr per phase or compute it.
            # Let's pass step count.
            
        return p, e, False

    # Let's rewrite the descent loop to be cleaner.
    
    def run_phase(pts, energy, steps, lr_start, lr_end):
        nonlocal best_energy, best_points
        p = pts
        e = energy
        for step in range(steps):
            if time.time() > deadline:
                return p, e, True
            
            # Geometric decay
            ratio = lr_end / lr_start
            lr = lr_start * (ratio ** (step / max(1, steps)))
            
            e, grad = _energy_and_grad(p)
            
            p = p - lr * grad
            p = p / np.linalg.norm(p, axis=1, keepdims=True)
            
            # Recalculate energy to be sure it's consistent with new p
            # (The e returned from _energy_and_grad(p) above is for the OLD p? 
            # No, _energy_and_grad takes p as input and returns energy of p.
            # So e is energy of p BEFORE update.
            # We need energy of p AFTER update.
            # So we must call _energy_and_grad(p) again or modify the function.
            # It is cheaper to just accept the step if we trust the gradient, 
            # but for tracking best, we need the true energy.
            # Let's call it again. It's O(N^2). 10k ops. Negligible.
            
            e_new, _ = _energy_and_grad(p)
            
            if e_new < best_energy:
                best_energy = e_new
                best_points = p.copy()
            
            e = e_new
            
        return p, e, False

    # Phase 1
    points, energy, timeout = run_phase(points, current_energy, 5000, 0.05, 0.001)
    
    # Phase 2
    if not timeout:
        points, energy, timeout = run_phase(points, energy, 5000, 0.001, 0.0001)
        
    # Phase 3
    if not timeout:
        points, energy, timeout = run_phase(points, energy, 5000, 0.0001, 0.00001)
        
    return best_points

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
