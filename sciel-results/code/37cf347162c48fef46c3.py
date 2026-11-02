"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = N
    # 1. Initialization: Vogel Spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(i / n)
        theta = i * golden_angle
        points[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    # Adam Optimizer State
    m = np.zeros_like(points)
    v = np.zeros_like(points)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.01
    
    # Parameters
    r_current = 0.02
    r_step = 0.0005
    max_r = 1.0 / np.sqrt(N) * 1.1
    time_budget = 110.0
    start_time = time.time()
    
    best_points = points.copy()
    best_score = 0.0
    
    def calc_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Initial score
    current_score = calc_score(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()
        
    def energy_and_grad(pts, r):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Use a slightly larger threshold to prevent pairs from getting too close
        # which causes numerical instability and local minima
        active_mask = dists < 2 * r * 1.05
        forces = np.zeros_like(pts)
        
        if np.any(active_mask):
            safe_dists = np.where(dists > 1e-8, dists, 1e-8)
            coeff = np.zeros_like(dists)
            # Steeper potential: (2r - d)^2 / d^2
            overlap_amount = 2 * r - dists[active_mask]
            coeff[active_mask] = (overlap_amount ** 2) / (dists[active_mask] ** 2)
            forces += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
        norms = np.linalg.norm(pts, axis=1)
        boundary_violation = norms - (1.0 - r)
        boundary_mask = boundary_violation > 0
        
        if np.any(boundary_mask):
            safe_norms = np.where(norms > 1e-8, norms, 1e-8)
            # Stronger boundary force
            f_boundary = -10.0 * boundary_violation[:, np.newaxis] * pts / safe_norms[:, np.newaxis]
            f_boundary[~boundary_mask] = 0
            forces += f_boundary
            
        # Energy for convergence check (use 4th power to be stricter)
        e_pair = np.sum((2 * r - dists[active_mask]) ** 4) if np.any(active_mask) else 0.0
        e_bound = np.sum(boundary_violation[boundary_mask] ** 2) if np.any(boundary_mask) else 0.0
        energy = e_pair + e_bound
        
        return energy, -forces

    t_step = 0
    while r_current < max_r and time.time() - start_time < time_budget:
        # Optimize positions for current r
        # Increase steps per radius increment to allow better convergence
        for step in range(1000):
            if time.time() - start_time > time_budget:
                break
            
            t_step += 1
            energy, grad = energy_and_grad(points, r_current)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            
            # Decay learning rate to allow fine-tuning
            current_lr = 0.01 * (0.999 ** (t_step // 100))
            
            points = points - current_lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
        current_score = calc_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        energy, _ = energy_and_grad(points, r_current)
        
        # Only increase radius if the packing is stable (low energy)
        if energy < 1e-5:
            r_current += r_step
        else:
            # If energy is high, don't increase radius much, add noise to escape local minima
            r_current += r_step * 0.01
            points += np.random.normal(0, 0.005, points.shape)

    return best_points
# EVOLVE-BLOCK-END
