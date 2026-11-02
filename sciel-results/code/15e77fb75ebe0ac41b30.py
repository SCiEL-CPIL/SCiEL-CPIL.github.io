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
        # Energy: Sum of squared violations
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        overlap_mask = dists < 2 * r
        forces = np.zeros_like(pts)
        if np.any(overlap_mask):
            safe_dists = np.where(dists > 1e-6, dists, 1e-6)
            coeff = np.zeros_like(dists)
            coeff[overlap_mask] = 2.0 * (2 * r - dists[overlap_mask]) / safe_dists[overlap_mask]
            forces += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
        norms = np.linalg.norm(pts, axis=1)
        boundary_violation = norms - (1.0 - r)
        boundary_mask = boundary_violation > 0
        
        if np.any(boundary_mask):
            safe_norms = np.where(norms > 1e-6, norms, 1e-6)
            f_boundary = -2.0 * boundary_violation[:, np.newaxis] * pts / safe_norms[:, np.newaxis]
            f_boundary[~boundary_mask] = 0
            forces += f_boundary
            
        e_pair = np.sum((2 * r - dists[overlap_mask]) ** 2) if np.any(overlap_mask) else 0.0
        e_bound = np.sum(boundary_violation[boundary_mask] ** 2) if np.any(boundary_mask) else 0.0
        energy = e_pair + e_bound
        
        return energy, -forces

    t_step = 0
    while r_current < max_r and time.time() - start_time < time_budget:
        # Optimize positions for current r
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
            
            t_step += 1
            energy, grad = energy_and_grad(points, r_current)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            
            points = points - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
        current_score = calc_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        energy, _ = energy_and_grad(points, r_current)
        
        if energy < 1e-4:
            r_current += r_step
        else:
            r_current += r_step * 0.1
            points += np.random.normal(0, 0.001, points.shape)

    return best_points
# EVOLVE-BLOCK-END
