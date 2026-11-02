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
    
    # Parameters
    # Estimate initial radius: approx 1/sqrt(N) is a lower bound, best is ~0.053
    # Start safely
    r_current = 0.04
    max_r = 0.06
    time_budget = 115.0
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
        
        forces = np.zeros_like(pts)
        
        # Pairwise overlap
        # We want dists >= 2*r
        # violation = max(0, 2*r - dists)
        # grad of sum(violation^2) w.r.t pts[i]
        overlap_mask = dists < 2 * r
        if np.any(overlap_mask):
            safe_dists = np.where(dists > 1e-6, dists, 1e-6)
            # coeff for pair (i,j) is 2*(2r - d) * (1/d) * (p_i - p_j)
            # Sum over j
            coeff = np.zeros_like(dists)
            coeff[overlap_mask] = 2.0 * (2 * r - dists[overlap_mask]) / safe_dists[overlap_mask]
            forces += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
        e_pair = np.sum((2 * r - dists[overlap_mask]) ** 2) if np.any(overlap_mask) else 0.0
        
        # Boundary violation
        # We want norms <= 1 - r
        # violation = max(0, norms - (1-r))
        norms = np.linalg.norm(pts, axis=1)
        boundary_violation = norms - (1.0 - r)
        boundary_mask = boundary_violation > 0
        
        if np.any(boundary_mask):
            safe_norms = np.where(norms > 1e-6, norms, 1e-6)
            # grad of (norm - (1-r))^2 = 2*(norm - (1-r)) * (p / norm)
            f_boundary = 2.0 * boundary_violation[:, np.newaxis] * pts / safe_norms[:, np.newaxis]
            f_boundary[~boundary_mask] = 0
            forces += f_boundary
            
        e_bound = np.sum(boundary_violation[boundary_mask] ** 2) if np.any(boundary_mask) else 0.0
        energy = e_pair + e_bound
        
        return energy, -forces

    t_step = 0
    r_step = 0.001
    lr_start = 0.05
    lr_end = 0.001
    
    # First, optimize for the initial r to settle the spiral
    for _ in range(200):
        t_step += 1
        energy, grad = energy_and_grad(points, r_current)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t_step)
        v_hat = v / (1 - beta2 ** t_step)
        lr = lr_start
        points = points - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
    current_score = calc_score(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()
        
    while r_current < max_r and time.time() - start_time < time_budget:
        # Adaptive learning rate: decay as we get closer to max_r or as energy drops
        # Simple heuristic: decay lr based on progress
        progress = (r_current - 0.04) / (max_r - 0.04)
        lr = lr_start * (1 - progress) + lr_end * progress
        
        # Optimize positions for current r
        # Use more iterations if we are making progress
        max_iters = 300
        success = False
        for step in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            t_step += 1
            energy, grad = energy_and_grad(points, r_current)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_step)
            v_hat = v / (1 - beta2 ** t_step)
            
            points = points - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            if energy < 1e-6:
                success = True
                break
        
        current_score = calc_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        energy, _ = energy_and_grad(points, r_current)
        
        if energy < 1e-5:
            r_current += r_step
        else:
            # If we failed to satisfy current r, don't increase. 
            # Maybe reduce r slightly or just retry with more iterations?
            # For now, small step and perturbation
            r_current += r_step * 0.1
            # Perturbation to escape local minima
            points += np.random.normal(0, 0.005, points.shape)
            # Reset momentum to avoid carrying over bad gradients
            m = np.zeros_like(points)
            v = np.zeros_like(points)
            t_step = 0

    return best_points
# EVOLVE-BLOCK-END
