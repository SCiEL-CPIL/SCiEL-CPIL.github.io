# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    deadline = time.time() + 590  # Leave 10s margin
    
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

    def coordinate_ascent(points, step_sizes, max_rounds=20):
        """Coordinate ascent with random directions and decreasing step sizes."""
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        for step_size in step_sizes:
            if time.time() >= deadline - 10:
                break
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for i in range(n):
                    if time.time() >= deadline - 10:
                        break
                    # Try moving point i in random directions
                    for _ in range(20):
                        if time.time() >= deadline - 10:
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

    def simulated_annealing(points, max_iter=2000):
        """Simulated annealing to escape local optima."""
        best_points = points.copy()
        best_score = compute_score(points)
        current_points = points.copy()
        current_score = best_score
        temperature = 0.05
        cooling_rate = 0.9995
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
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
    
    # 4. Hexagonal lattice subset
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:16])
    config4 = normalize(hex_pts)
    initial_configs.append(config4)
    
    # 5. Random with seed
    np.random.seed(42)
    config5 = np.random.randn(n, 2)
    config5 = normalize(config5)
    initial_configs.append(config5)

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
        
        # Phase 2: Simulated Annealing from coordinate ascent result
        if time.time() < deadline - 20:
            sa_points, sa_score = simulated_annealing(ca_points, max_iter=1000)
            if sa_score > best_overall_score:
                best_overall_score = sa_score
                best_overall = sa_points.copy()

    # Phase 3: Differential Evolution for global search
    # Use a smaller population and fewer iterations to fit in time
    if time.time() < deadline - 60:
        try:
            bounds = [(-5.0, 5.0)] * (n * 2)
            
            # Use the best found so far as a starting hint? DE doesn't use initial points directly in this way.
            # But we can run DE.
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

    # Phase 4: Final Refinement with Nelder-Mead on the best candidate
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
