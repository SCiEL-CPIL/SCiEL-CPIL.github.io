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

    def normalize(points):
        """Center and scale points to unit radius."""
        center = np.mean(points, axis=0)
        points = points - center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 1e-15:
            points = points / max_dist
        return points

    def coordinate_ascent(points, step_sizes, max_rounds=40):
        """
        Coordinate ascent: for each point, try random directions with given step size.
        Accept if score improves. Repeat for multiple step sizes (coarse to fine).
        """
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        for step_size in step_sizes:
            if time.time() >= deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(n):
                    if time.time() >= deadline - 5:
                        break
                    # Try multiple random directions for point i
                    for _ in range(20):
                        if time.time() >= deadline - 5:
                            break
                        direction = np.random.randn(2)
                        norm = np.linalg.norm(direction)
                        if norm < 1e-10:
                            continue
                        direction /= norm
                        
                        new_points = best_points.copy()
                        new_points[i] += direction * step_size
                        score = compute_score(new_points)
                        if score > best_score:
                            best_points = new_points
                            best_score = score
                            improved = True
                            break
        return best_points, best_score

    def gradient_ascent(points, lr=0.005, steps=800):
        """
        Gradient ascent using finite differences.
        Compute gradient of score w.r.t. each coordinate via central differences.
        """
        best_points = points.copy()
        best_score = compute_score(best_points)
        current_points = best_points.copy()
        current_score = best_score
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient via central differences
            grad = np.zeros_like(current_points)
            eps = 1e-6
            for i in range(n):
                for j in range(2):
                    perturbed_plus = current_points.copy()
                    perturbed_plus[i, j] += eps
                    f_plus = compute_score(perturbed_plus)
                    
                    perturbed_minus = current_points.copy()
                    perturbed_minus[i, j] -= eps
                    f_minus = compute_score(perturbed_minus)
                    
                    if np.isfinite(f_plus) and np.isfinite(f_minus):
                        grad[i, j] = (f_plus - f_minus) / (2 * eps)
                    else:
                        grad[i, j] = 0.0
            
            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Update points with adaptive learning rate
            effective_lr = lr * (1.0 + 0.5 * np.sin(step * 0.01))  # slight oscillation
            new_points = current_points + effective_lr * grad
            new_points = normalize(new_points)
            
            new_score = compute_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            else:
                # If no improvement, reduce learning rate and try again
                lr *= 0.95
            
            # Decay learning rate
            lr = max(lr * 0.9995, 0.0001)
        
        return best_points, best_score

    def simulated_annealing(points, max_iter=2000):
        """Simulated annealing to escape local optima."""
        best_points = points.copy()
        best_score = compute_score(points)
        current_points = points.copy()
        current_score = best_score
        temperature = 0.08
        cooling_rate = 0.9997
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation with Gaussian noise
            perturbation = np.random.randn(n, 2) * temperature
            perturbed = current_points + perturbation
            perturbed = normalize(perturbed)
            
            new_score = compute_score(perturbed)
            
            # Accept if better, or with probability based on temperature
            if new_score >= current_score:
                current_points = perturbed
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = perturbed.copy()
            elif temperature > 1e-8:
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / temperature):
                    current_points = perturbed
                    current_score = new_score
            
            temperature *= cooling_rate
        
        return best_points, best_score

    def refine(points):
        """Full refinement pipeline: coordinate ascent → gradient → SA → coordinate → gradient."""
        # Phase 1: Coordinate ascent (coarse to fine)
        step_sizes = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
        pts, sc = coordinate_ascent(points, step_sizes, max_rounds=30)
        
        if time.time() > deadline - 30:
            return pts, sc
        
        # Phase 2: Gradient ascent
        pts, sc = gradient_ascent(pts, lr=0.005, steps=600)
        
        if time.time() > deadline - 30:
            return pts, sc
        
        # Phase 3: Simulated annealing to escape local optima
        pts, sc = simulated_annealing(pts, max_iter=1500)
        
        if time.time() > deadline - 20:
            return pts, sc
        
        # Phase 4: Final coordinate ascent + gradient for polish
        step_sizes_fine = [0.03, 0.015, 0.008, 0.004, 0.002]
        pts, sc = coordinate_ascent(pts, step_sizes_fine, max_rounds=25)
        
        if time.time() > deadline - 15:
            return pts, sc
        
        pts, sc = gradient_ascent(pts, lr=0.003, steps=400)
        
        return pts, sc

    # Generate diverse initial configurations
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
    config2 = normalize(config2)
    initial_configs.append(config2)
    
    # 3. Two concentric rings (8 inner, 8 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = 0.55 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config3 = np.vstack([inner_pts, outer_pts])
    config3 = normalize(config3)
    initial_configs.append(config3)
    
    # 4. Three concentric rings (6, 5, 5)
    r1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    r3_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r1_pts = 0.35 * np.column_stack([np.cos(r1_angles), np.sin(r1_angles)])
    r2_pts = 0.65 * np.column_stack([np.cos(r2_angles), np.sin(r2_angles)])
    r3_pts = 1.0 * np.column_stack([np.cos(r3_angles), np.sin(r3_angles)])
    config4 = np.vstack([r1_pts, r2_pts, r3_pts])
    config4 = normalize(config4)
    initial_configs.append(config4)
    
    # 5. Hexagonal lattice subset (center + 6 + 9)
    hex_pts = []
    hex_pts.append([0.0, 0.0])
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for a in hex_angles:
        hex_pts.append([np.cos(a), np.sin(a)])
    hex_angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for a in hex_angles2[:9]:
        hex_pts.append([1.732 * np.cos(a), 1.732 * np.sin(a)])
    config5 = np.array(hex_pts[:16])
    config5 = normalize(config5)
    initial_configs.append(config5)
    
    # 6. Two offset squares (4+4+4+4 = 16)
    sq1 = np.array([[-0.7, -0.7], [-0.7, 0.7], [0.7, -0.7], [0.7, 0.7]])
    sq2 = np.array([[-1.2, 0.0], [0.0, -1.2], [1.2, 0.0], [0.0, 1.2]])
    sq3 = np.array([[-0.4, 0.0], [0.4, 0.0], [0.0, -0.4], [0.0, 0.4]])
    sq4 = np.array([[-1.0, -0.5], [-0.5, 1.0], [1.0, 0.5], [0.5, -1.0]])
    config6 = np.vstack([sq1, sq2, sq3, sq4])
    config6 = normalize(config6)
    initial_configs.append(config6)
    
    # 7. Random with fixed seed
    np.random.seed(42)
    config7 = np.random.randn(n, 2)
    config7 = normalize(config7)
    initial_configs.append(config7)
    
    # 8. Another random
    np.random.seed(123)
    config8 = np.random.randn(n, 2)
    config8 = normalize(config8)
    initial_configs.append(config8)
    
    # Optimize each configuration
    best_overall = None
    best_overall_score = -np.inf
    
    # Allocate time budget: more time for promising configs
    total_configs = len(initial_configs)
    time_per_config = (deadline - start_time - 30) / total_configs
    
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        # Set a sub-deadline for this config
        config_deadline = min(deadline - 15, start_time + (i + 1) * time_per_config)
        
        # Temporarily adjust deadline for this config's refinement
        global_deadline_backup = deadline
        deadline = config_deadline
        
        try:
            opt_points, opt_score = refine(config)
        finally:
            deadline = global_deadline_backup
        
        if opt_score > best_overall_score:
            best_overall_score = opt_score
            best_overall = opt_points.copy()
    
    # Final global refinement from the best found
    if best_overall is not None and time.time() < deadline - 20:
        final_points, final_score = refine(best_overall)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall = final_points.copy()
    
    # Final normalization
    if best_overall is not None:
        best_overall = normalize(best_overall)
    
    # Fallback to regular polygon if something went wrong
    if best_overall is None or not np.all(np.isfinite(best_overall)):
        best_overall = config1
    
    return best_overall


# EVOLVE-BLOCK-END
