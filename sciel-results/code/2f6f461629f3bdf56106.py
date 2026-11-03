"""Improved packing: SGLD-based energy minimization with radius increase."""
import numpy as np
import time

N = 100

def _compute_score(centers):
    """Compute the score (radius) for a set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check boundaries
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Compute wall distances
    wall_dists = np.min(np.stack([
        centers[:, 0] + 0.5,
        0.5 - centers[:, 0],
        centers[:, 1] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 2] + 0.5,
        0.5 - centers[:, 2]
    ]), axis=0)
    min_wall = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall)


def _energy_and_grad(centers, r):
    """
    Compute energy (sum of squared overlaps) and gradient.
    Energy = sum over pairs of max(0, r*2 - dist)^2 + sum over walls of max(0, r - wall_dist)^2
    We want to minimize this to make all constraints satisfied with radius r.
    """
    n = centers.shape[0]
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise interactions
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 2*r and dist > 1e-12:
                overlap = 2*r - dist
                # d(overlap^2)/d(centers[i]) = 2*overlap * d(overlap)/d(centers[i])
                # d(overlap)/d(centers[i]) = -diff/dist
                grad[i] += 2 * overlap * (-diff / dist)
                grad[j] += 2 * overlap * (diff / dist)
                energy += overlap**2
    
    # Wall interactions
    for i in range(n):
        for axis in range(3):
            # Lower wall: center[axis] + 0.5
            wall_dist_lower = centers[i, axis] + 0.5
            if wall_dist_lower < r:
                overlap = r - wall_dist_lower
                grad[i, axis] += 2 * overlap * (-1.0)  # d(overlap)/d(centers[i,axis]) = -1
                energy += overlap**2
            # Upper wall: 0.5 - center[axis]
            wall_dist_upper = 0.5 - centers[i, axis]
            if wall_dist_upper < r:
                overlap = r - wall_dist_upper
                grad[i, axis] += 2 * overlap * (1.0)  # d(overlap)/d(centers[i,axis]) = -1, but overlap = r - (0.5 - c) = r - 0.5 + c, so d/dc = 1
                energy += overlap**2
    
    return energy, grad


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Initialize with a grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # SGLD parameters
    lr = 0.001
    noise_scale = 0.005
    r = best_score * 0.95  # Start slightly below current feasible radius
    
    # Phase 1: SGLD to find a good configuration
    n_iter = 5000
    for it in range(n_iter):
        if time.time() > deadline - 20.0:
            break
        
        # Compute energy and gradient at current radius
        energy, grad = _energy_and_grad(centers, r)
        
        # SGLD update: center -= lr * grad + noise
        noise = noise_scale * np.random.randn(*centers.shape)
        centers = centers - lr * grad + noise
        
        # Project back into cube (with small margin)
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Try to increase radius periodically
        if it % 100 == 99:
            current_score = _compute_score(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
            # Increase r if we're doing well
            if energy < 1e-4:
                r = min(r * 1.005, 0.112)
            # Decay noise and lr
            noise_scale *= 0.995
            lr *= 0.999
    
    # Phase 2: Local refinement with L-BFGS-B on the best configuration
    if time.time() < deadline - 15.0:
        from scipy.optimize import minimize
        
        def objective(x):
            c = x.reshape(N, 3)
            if np.any(c < -0.5 + 1e-9) or np.any(c > 0.5 - 1e-9):
                return 1e10
            return -_compute_score(c)
        
        # Try a few different starting points
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            
            if attempt == 0:
                x0 = best_centers.flatten()
            else:
                x0 = best_centers.flatten() + 0.001 * np.random.randn(N * 3)
                x0 = np.clip(x0.reshape(N, 3), -0.5 + 0.001, 0.5 - 0.001).flatten()
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success and res.fun < -best_score:
                new_centers = res.x.reshape(N, 3)
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers
    
    # Final polish: very fine local search
    if time.time() < deadline - 5.0:
        from scipy.optimize import minimize
        
        def objective_fine(x):
            c = x.reshape(N, 3)
            if np.any(c < -0.5 + 1e-9) or np.any(c > 0.5 - 1e-9):
                return 1e10
            return -_compute_score(c)
        
        res = minimize(objective_fine, best_centers.flatten(), method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
        
        if res.fun < -best_score:
            new_centers = res.x.reshape(N, 3)
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
    
    # Ensure final result is valid
    final_score = _compute_score(best_centers)
    if final_score <= 0:
        # Fallback to grid
        return grid[:N]
    
    return best_centers
