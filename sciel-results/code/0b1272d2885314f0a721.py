# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    n = 16
    d = 2
    
    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """Exact objective for final evaluation."""
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
            
        points = params.reshape(n, d)
        R = compute_R(points)
        return -R  # minimize -R

    def simulated_annealing(initial_points, n_iter=50000, t_start=1.0, t_end=0.001, seed=42):
        """
        Simulated Annealing to explore the global landscape.
        """
        rng = np.random.RandomState(seed)
        current_points = initial_points.copy()
        current_R = compute_R(current_points)
        best_points = current_points.copy()
        best_R = current_R
        
        n_iter = min(n_iter, 200000)  # Cap iterations
        
        for i in range(n_iter):
            # Check deadline
            if time.time() > start_time + deadline - safety_margin:
                break
            
            # Cooling schedule
            alpha = i / n_iter
            temperature = t_start * (t_end / t_start) ** alpha
            
            # Generate perturbation: move one or two points
            n_moves = 1 if rng.rand() < 0.7 else 2
            indices = rng.choice(n, size=n_moves, replace=False)
            
            # Perturbation size decreases with temperature
            scale = 0.5 * temperature + 0.01
            new_points = current_points.copy()
            for idx in indices:
                new_points[idx] += rng.normal(0, scale, size=d)
            
            new_R = compute_R(new_points)
            
            # Acceptance criterion
            if new_R > current_R or rng.rand() < np.exp((new_R - current_R) / max(temperature, 1e-10)):
                current_points = new_points
                current_R = new_R
                if current_R > best_R:
                    best_R = current_R
                    best_points = current_points.copy()
        
        return best_points, best_R

    def local_refine(points):
        """
        Local refinement using Powell and Nelder-Mead.
        """
        if time.time() > start_time + deadline - safety_margin:
            return points
        
        params = points.flatten()
        best_params = params.copy()
        best_val = objective(params)
        
        # Powell refinement
        try:
            result = minimize(
                objective, 
                best_params, 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        # Nelder-Mead polish
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        return best_params.reshape(n, d)

    # Generate diverse initial points for SA
    def make_initial_points(seed):
        """Create various initial configurations."""
        rng = np.random.RandomState(seed)
        config_type = seed % 4
        
        if config_type == 0:
            # 4x4 grid
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.ravel(), Y.ravel()))
        elif config_type == 1:
            # Random uniform in unit square
            points = rng.uniform(-1, 1, size=(n, d))
        elif config_type == 2:
            # Regular hexagonal packing approximation
            points = np.zeros((n, d))
            for i in range(n):
                row = i // 4
                col = i % 4
                points[i, 0] = col * 1.0 + (row % 2) * 0.5
                points[i, 1] = row * 0.866
            points -= points.mean(axis=0)
        else:
            # Random on circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radii = rng.uniform(0.5, 1.5, size=n)
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        return points

    # Run multiple SA restarts with different initial conditions
    best_points = None
    best_R = 0.0
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > start_time + deadline - safety_margin:
            break
        
        init_points = make_initial_points(seed=42 + restart)
        
        # Run SA with this initial condition
        sa_points, sa_R = simulated_annealing(init_points, n_iter=30000, t_start=1.0, t_end=0.001, seed=42 + restart)
        
        if sa_R > best_R:
            best_R = sa_R
            best_points = sa_points.copy()
        
        # Local refinement
        refined_points = local_refine(sa_points)
        refined_R = compute_R(refined_points)
        
        if refined_R > best_R:
            best_R = refined_R
            best_points = refined_points.copy()
    
    # Final high-precision polish on the best result
    if best_points is not None:
        if time.time() < start_time + deadline - safety_margin:
            final_points = local_refine(best_points)
            final_R = compute_R(final_points)
            if final_R > best_R:
                best_points = final_points
                best_R = final_R
    
    # Fallback if something went wrong
    if best_points is None or best_points.shape != (n, d):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        best_points = np.column_stack((X.ravel(), Y.ravel()))
    
    return best_points

# EVOLVE-BLOCK-END
