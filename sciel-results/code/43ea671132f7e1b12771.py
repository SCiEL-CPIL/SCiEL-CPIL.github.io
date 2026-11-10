# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # Leave 30s margin from 630s

    # Precompute indices for all triples of points
    idx = np.array(list((i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)))
    n_triplets = len(idx)

    def all_triangle_areas(points):
        """
        Calculate all triangle areas among all triples of points using vectorization.
        points: (n, 2)
        Returns: array of shape (286,)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return areas

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            # Add a small epsilon to avoid degeneracy if points are nearly collinear
            # though ConvexHull handles it, we want to be safe
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def objective(points_t, beta=1000.0):
        """
        Objective function for optimization.
        We minimize -A_min / A_hull.
        To handle the non-smooth min, we use a soft-min for the optimization steps,
        but we can switch to exact min for final evaluation or if beta is very high.
        For DE, a moderate beta helps smoothness. For local polish, high beta.
        """
        points = points_t.reshape(n, 2)
        
        # Check for validity
        if not np.all(np.isfinite(points)):
            return 1e10
            
        areas = all_triangle_areas(points)
        
        # Soft min approximation for smoothness during global search
        # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        # If beta is large, it approaches min(x)
        if beta < 1e6:
            shifted = -beta * areas
            max_shifted = np.max(shifted)
            # Handle potential overflow/underflow
            log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            A_min_smooth = -log_sum / beta
        else:
            A_min_smooth = np.min(areas)
            
        A_hull = hull_area(points)
        
        if A_hull < 1e-10 or A_min_smooth < 1e-10:
            return 1e10
            
        return -A_min_smooth / A_hull

    def exact_score(points):
        """Calculate the actual score: A_min / A_hull"""
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        A_min = np.min(all_triangle_areas(points))
        if A_min < 1e-12:
            return 0.0
        return A_min / A_hull

    # Define bounds for differential evolution
    # We can fix the scale and rotation to reduce dimensionality, 
    # but for simplicity and robustness, we optimize in 26D with bounds [-1, 1].
    # The objective is scale-invariant (ratio), so the optimizer will find a stable scale.
    # However, DE works better with bounded spaces.
    bounds = [(-1.0, 1.0)] * (n * 2)

    # Generate initial population manually to guide DE towards good structures
    # DE accepts a 'init' parameter. We can provide a population array.
    # Shape: (pop_size, 2*n)
    
    pop_size = 20
    initial_pop = np.zeros((pop_size, n * 2))
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_pop[0] = reg_poly.flatten()
    
    # 2. Concentric rings (common optimal structure for Heilbronn)
    # e.g., 7 on outer, 6 on inner? Or 6 outer, 6 inner, 1 center?
    # Let's try a few variations
    
    # Variation A: 1 center, 6 inner, 6 outer
    c1 = np.array([0.0, 0.0])
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.6 * np.cos(inner_angles), 0.6 * np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
    outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
    config_a = np.vstack([c1, inner_pts, outer_pts])
    initial_pop[1] = config_a.flatten()
    
    # Variation B: 2 center-ish, 11 outer? No, let's stick to rings.
    # Variation C: 7 outer, 6 inner
    outer_angles7 = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts7 = np.column_stack([1.0 * np.cos(outer_angles7), 1.0 * np.sin(outer_angles7)])
    inner_angles6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_pts6 = np.column_stack([0.5 * np.cos(inner_angles6), 0.5 * np.sin(inner_angles6)])
    config_c = np.vstack([inner_pts6, outer_pts7])
    initial_pop[2] = config_c.flatten()
    
    # 3. Random perturbations of regular polygon
    rng = np.random.default_rng(seed=42)
    for i in range(3, pop_size):
        perturbation = rng.normal(0, 0.1, (n, 2))
        pts = reg_poly + perturbation
        initial_pop[i] = pts.flatten()

    # Run Differential Evolution
    # strategy='best1bin' is robust
    # maxiter is kept moderate to allow time for local polish
    # tol is tight
    try:
        result_de = differential_evolution(
            lambda x: objective(x, beta=50.0), 
            bounds, 
            init=initial_pop,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False, # We will do our own polish
            disp=False
        )
        
        best_points = result_de.x.reshape(n, 2)
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to regular polygon
        best_points = reg_poly

    # Local Polish Phase
    # Use Nelder-Mead on the exact (or high-beta soft) objective to refine the best point.
    # We will try a few different beta values for the local polish to navigate the non-smooth landscape.
    
    current_best_points = best_points
    current_best_score = exact_score(current_best_points)
    
    # We can also try to "snap" points to the convex hull if they are very close, 
    # but usually, optimal points are either on the hull or strictly inside.
    # Let's just run local minimization.
    
    # To handle the non-differentiability of min(), we use a high beta soft-min for the gradient-free NM.
    # Or we can just use the exact min, as NM is derivative-free and can handle non-smooth functions better than gradient-based methods.
    
    def objective_local(points_t):
        return objective(points_t, beta=1000.0) # High beta for near-exact min

    # Try multiple local optimizations with different starting perturbations or simplex sizes
    for simplex_size in [0.05, 0.01, 0.001]:
        if time.time() - start_time > time_limit - 60:
            break
            
        try:
            result_nm = minimize(
                objective_local,
                current_best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'initial_simplex': simplex_size
                }
            )
            
            if result_nm.success:
                candidate_points = result_nm.x.reshape(n, 2)
                candidate_score = exact_score(candidate_points)
                
                if candidate_score > current_best_score:
                    current_best_score = candidate_score
                    current_best_points = candidate_points
                    
        except Exception as e:
            pass

    # Final verification
    if not np.all(np.isfinite(current_best_points)):
        current_best_points = reg_poly

    return current_best_points

# EVOLVE-BLOCK-END
