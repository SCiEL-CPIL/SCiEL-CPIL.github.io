# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin of 10 seconds

    n = 16

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective(points):
        """Negative of the score for minimization."""
        score = compute_score(points)
        if np.isinf(score) or np.isnan(score):
            return 1e10
        return -score

    def gradient_ascent(points, lr=0.01, steps=1000):
        """Simple gradient ascent using finite differences on the score."""
        best_points = points.copy()
        best_score = compute_score(points)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            # Compute gradient via central differences
            grad = np.zeros_like(points)
            eps = 1e-5
            for i in range(n):
                for j in range(2):
                    perturbed = points.copy()
                    perturbed[i, j] += eps
                    f_plus = compute_score(perturbed)
                    
                    perturbed = points.copy()
                    perturbed[i, j] -= eps
                    f_minus = compute_score(perturbed)
                    
                    if np.isfinite(f_plus) and np.isfinite(f_minus):
                        grad[i, j] = (f_plus - f_minus) / (2 * eps)
                    else:
                        grad[i, j] = 0.0
            
            # Update points
            points = points + lr * grad
            
            # Normalize to keep points in reasonable range
            center = np.mean(points, axis=0)
            points -= center
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points /= max_dist
            
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Decay learning rate
            lr *= 0.999
        
        return best_points, best_score

    def simulated_annealing(points, max_iter=5000):
        """Simulated annealing to escape local optima."""
        best_points = points.copy()
        best_score = compute_score(points)
        current_points = points.copy()
        current_score = best_score
        temperature = 0.1
        cooling_rate = 0.999
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Random perturbation
            perturbed = current_points + np.random.randn(n, 2) * temperature
            
            # Normalize
            center = np.mean(perturbed, axis=0)
            perturbed -= center
            max_dist = np.max(np.linalg.norm(perturbed, axis=1))
            if max_dist > 0:
                perturbed /= max_dist
            
            new_score = compute_score(perturbed)
            
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / temperature):
                current_points = perturbed
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = perturbed.copy()
            
            temperature *= cooling_rate
        
        return best_points, best_score

    # Generate initial configurations
    initial_configs = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    config1 = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(config1)
    
    # 2. Square grid 4x4
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_x, grid_y = np.meshgrid(xs, ys)
    config2 = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    initial_configs.append(config2)
    
    # 3. Two concentric squares
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config3 = np.vstack([inner_pts, outer_pts])
    initial_configs.append(config3)
    
    # 4. Random with seed
    np.random.seed(42)
    config4 = np.random.randn(n, 2)
    config4 -= np.mean(config4, axis=0)
    max_d = np.max(np.linalg.norm(config4, axis=1))
    if max_d > 0:
        config4 /= max_d
    initial_configs.append(config4)
    
    # 5. Hexagonal-ish pattern
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_pts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    # Add 10 more points in a ring
    ring_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    ring_pts = 1.5 * np.column_stack([np.cos(ring_angles), np.sin(ring_angles)])
    config5 = np.vstack([hex_pts, ring_pts])
    initial_configs.append(config5)
    
    # Optimize each configuration
    best_overall = None
    best_overall_score = -np.inf
    
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        # Gradient ascent
        opt_points, opt_score = gradient_ascent(config, lr=0.05, steps=500)
        
        if opt_score > best_overall_score:
            best_overall_score = opt_score
            best_overall = opt_points.copy()
        
        # Simulated annealing from the gradient result
        sa_points, sa_score = simulated_annealing(opt_points, max_iter=2000)
        
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall = sa_points.copy()
        
        # Also try SA from original config
        sa2_points, sa2_score = simulated_annealing(config, max_iter=2000)
        
        if sa2_score > best_overall_score:
            best_overall_score = sa2_score
            best_overall = sa2_points.copy()
        
        # Final gradient polish
        if time.time() < deadline - 10:
            final_points, final_score = gradient_ascent(best_overall, lr=0.01, steps=300)
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall = final_points.copy()
    
    # Final normalization
    if best_overall is not None:
        center = np.mean(best_overall, axis=0)
        best_overall -= center
        max_dist = np.max(np.linalg.norm(best_overall, axis=1))
        if max_dist > 0:
            best_overall /= max_dist
    
    # Fallback to regular polygon if something went wrong
    if best_overall is None or not np.all(np.isfinite(best_overall)):
        best_overall = config1
    
    return best_overall


# EVOLVE-BLOCK-END
