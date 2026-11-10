# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Leave 30s margin for safety

    def compute_score(points):
        """Compute R = (d_min / d_max)^2"""
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

    def objective(flat_points):
        """Objective for minimization: minimize d_max / d_min (equivalent to maximizing d_min/d_max)"""
        points = flat_points.reshape(n, 2)
        d = pdist(points, metric='euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min < 1e-10:
            return 1e10
        return (d_max / d_min)

    def normalize(points):
        """Center and scale points to unit radius."""
        center = np.mean(points, axis=0)
        points = points - center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points

    def coordinate_ascent(points, step_sizes, max_rounds=50):
        """Coordinate ascent with random directions and decreasing step sizes."""
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
                    # Try moving point i in random directions
                    for _ in range(30):
                        if time.time() >= deadline - 5:
                            break
                        direction = np.random.randn(2)
                        norm = np.linalg.norm(direction)
                        if norm > 0:
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

    def gradient_ascent(points, lr=0.01, steps=1000):
        """Gradient ascent using finite differences."""
        best_points = points.copy()
        best_score = compute_score(points)
        
        for step in range(steps):
            if time.time() > deadline - 10:
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
            
            # Normalize
            points = normalize(points)
            
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Decay learning rate
            lr *= 0.999
        
        return best_points, best_score

    def simulated_annealing(points, max_iter=3000):
        """Simulated annealing to escape local optima."""
        best_points = points.copy()
        best_score = compute_score(points)
        current_points = points.copy()
        current_score = best_score
        temperature = 0.1
        cooling_rate = 0.9995
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            perturbed = current_points + np.random.randn(n, 2) * temperature
            
            # Normalize
            perturbed = normalize(perturbed)
            
            new_score = compute_score(perturbed)
            
            if new_score > current_score or (temperature > 1e-6 and np.random.random() < np.exp((new_score - current_score) / temperature)):
                current_points = perturbed
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = perturbed.copy()
            
            temperature *= cooling_rate
        
        return best_points, best_score

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
    inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config3 = np.vstack([inner_pts, outer_pts])
    config3 = normalize(config3)
    initial_configs.append(config3)
    
    # 4. Three concentric rings (6, 5, 5)
    ring1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    ring3_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r1_pts = 0.4 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    r2_pts = 0.7 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    r3_pts = 1.0 * np.column_stack([np.cos(ring3_angles), np.sin(ring3_angles)])
    config4 = np.vstack([r1_pts, r2_pts, r3_pts])
    config4 = normalize(config4)
    initial_configs.append(config4)
    
    # 5. Random with seed
    np.random.seed(42)
    config5 = np.random.randn(n, 2)
    config5 = normalize(config5)
    initial_configs.append(config5)
    
    # 6. Hexagonal lattice subset
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:16])
    config6 = normalize(hex_pts)
    initial_configs.append(config6)

    # 7. Radial distortion of 4x4 grid (inspiration: better initialization)
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    # Apply radial distortion: push corners out, pull edges in slightly
    norms = np.linalg.norm(grid_pts, axis=1)
    max_norm = np.max(norms)
    # Non-linear scaling: r' = r^1.2 normalized
    radial_scale = (norms / max_norm) ** 1.2
    grid_pts_distorted = grid_pts * (radial_scale / norms[:, np.newaxis])
    config7 = normalize(grid_pts_distorted)
    initial_configs.append(config7)

    # 8. More varied concentric rings with phase shifts
    # 4 rings of 4 points each
    ring_configs = []
    for k in range(4):
        r = 0.3 + 0.25 * k
        phase = np.pi / 8 * k
        angles_k = np.linspace(0, 2 * np.pi, 4, endpoint=False) + phase
        ring_configs.append(r * np.column_stack([np.cos(angles_k), np.sin(angles_k)]))
    config8 = np.vstack(ring_configs)
    config8 = normalize(config8)
    initial_configs.append(config8)

    # 9. 5-5-6 ring structure
    r1_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    r3_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r1_pts = 0.3 * np.column_stack([np.cos(r1_angles), np.sin(r1_angles)])
    r2_pts = 0.6 * np.column_stack([np.cos(r2_angles), np.sin(r2_angles)])
    r3_pts = 1.0 * np.column_stack([np.cos(r3_angles), np.sin(r3_angles)])
    config9 = np.vstack([r1_pts, r2_pts, r3_pts])
    config9 = normalize(config9)
    initial_configs.append(config9)

    best_overall = None
    best_overall_score = -np.inf

    # Process each initial configuration
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Coordinate ascent
        ca_points, ca_score = coordinate_ascent(config, [0.1, 0.05, 0.02, 0.01, 0.005])
        
        if ca_score > best_overall_score:
            best_overall_score = ca_score
            best_overall = ca_points.copy()
        
        # Phase 2: Gradient ascent from coordinate ascent result
        if time.time() < deadline - 15:
            ga_points, ga_score = gradient_ascent(ca_points, lr=0.01, steps=500)
            if ga_score > best_overall_score:
                best_overall_score = ga_score
                best_overall = ga_points.copy()

        # Phase 3: Simulated Annealing from best of above
        if time.time() < deadline - 10:
            sa_points, sa_score = simulated_annealing(best_overall, max_iter=1500)
            if sa_score > best_overall_score:
                best_overall_score = sa_score
                best_overall = sa_points.copy()

    # Phase 4: Differential Evolution for global search
    # Use a smaller population and fewer iterations to fit in time
    if time.time() < deadline - 60:
        try:
            bounds = [(-5.0, 5.0)] * (n * 2)
            
            res = differential_evolution(
                objective, 
                bounds=bounds, 
                seed=42, 
                maxiter=50, # Reduced iterations
                popsize=8,   # Reduced population
                tol=1e-6, 
                init='latinhypercube'
            )
            
            if res.success:
                de_points = res.x.reshape(n, 2)
                de_score = compute_score(de_points)
                if de_score > best_overall_score:
                    best_overall_score = de_score
                    best_overall = de_points.copy()
                    
        except Exception:
            pass

    # Phase 5: Final Refinement with Nelder-Mead on the best candidate
    if best_overall is not None and time.time() < deadline - 5:
        try:
            start_point = best_overall.flatten()
            res_nm = minimize(
                objective, 
                start_point, 
                method='Nelder-Mead', 
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500}
            )
            
            if res_nm.fun < objective(start_point):
                if np.all(np.isfinite(res_nm.x)):
                    nm_points = res_nm.x.reshape(n, 2)
                    nm_score = compute_score(nm_points)
                    if nm_score > best_overall_score:
                        best_overall = nm_points
                        best_overall_score = nm_score
        except Exception:
            pass

    # Final safety check and return
    if best_overall is None:
        # Fallback to a regular polygon if everything failed
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_overall = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Normalize the points? The problem doesn't require a specific scale.
    # R = (d_min/d_max)^2 is scale invariant.
    # So any scale is fine.
    # However, floating point precision might be better if centered at 0.
    pts_final = best_overall.copy()
    pts_final -= np.mean(pts_final, axis=0)
    
    # Ensure it's a valid numpy array
    pts_final = np.array(pts_final, dtype=np.float64)
    
    return pts_final

# EVOLVE-BLOCK-END
