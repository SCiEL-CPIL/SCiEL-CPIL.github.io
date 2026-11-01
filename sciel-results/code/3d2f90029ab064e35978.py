import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_min_area(pts):
        """Compute minimum triangle area and hull area"""
        areas = []
        for i, j, k in itertools.combinations(range(N), 3):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas.append(0.5 * abs(det))
        min_area = min(areas)
        
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            hull_area = 1.0
        
        if hull_area < 1e-10:
            return 0.0, 1.0
        return min_area, hull_area
    
    def smoothed_objective(x, beta):
        """
        Smoothed objective using log-sum-exp approximation of min.
        We want to maximize min_area / hull_area.
        
        Using log-sum-exp: min(a_i) ≈ -1/beta * log(sum(exp(-beta * a_i)))
        This gives a smooth lower bound on the minimum.
        """
        pts = get_points(x)
        
        # Compute all triangle areas
        areas = []
        for i, j, k in itertools.combinations(range(N), 3):
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            areas.append(0.5 * abs(det))
        
        areas = np.array(areas)
        
        # Log-sum-exp for smooth min
        # min(areas) ≈ -1/beta * log(sum(exp(-beta * areas)))
        max_area = np.max(areas)
        log_sum_exp = max_area + np.log(np.sum(np.exp(-beta * (areas - max_area))))
        smooth_min = -log_sum_exp / beta
        
        # Hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 1.0  # Return high value (we're minimizing) for invalid config
        
        if hull_area < 1e-10:
            return 1.0
        
        # We want to maximize smooth_min / hull_area, so minimize the negative
        return -(smooth_min / hull_area)
    
    def raw_objective(x):
        """Raw objective for final evaluation"""
        pts = get_points(x)
        min_area, hull_area = compute_min_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)
    
    # Initial guess: points roughly on a circle
    def initial_guess(seed_offset=0.0, radius=1.0, center_x=0.5, center_y=0.5):
        x = np.zeros(2 * (N - 2))
        # Place points on a circle, skipping first two
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = center_x + radius * np.cos(angles[idx])
            x[2 * i + 1] = center_y + radius * np.sin(angles[idx])
        return x
    
    # Time limit management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_score = -np.inf
    best_x = None
    
    # Phase 1: Global search with differential evolution
    bounds = [(-3.0, 3.0) for _ in range(2 * (N - 2))]
    
    try:
        result_de = differential_evolution(
            raw_objective,
            bounds,
            seed=42,
            maxiter=30,
            popsize=10,
            tol=1e-6,
            init='sobol'
        )
        best_x = result_de.x
        best_score = -result_de.fun
    except Exception:
        pass
    
    if best_x is None:
        # Fallback to circular arrangement
        best_x = initial_guess(0.0)
        best_score = -raw_objective(best_x)
    
    # Phase 2: Multi-scale log-sum-exp refinement
    # Use progressively higher beta values for finer resolution
    beta_schedule = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        
        # Multiple diverse restarts from different configurations
        n_restarts = 4
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate diverse starting points
            if restart == 0:
                # Start from current best
                x0 = best_x.copy()
            elif restart == 1:
                # Regular polygon on circle
                x0 = initial_guess(0.3 * restart, radius=0.8 + 0.1 * restart)
            elif restart == 2:
                # Slightly perturbed circle
                x0 = initial_guess(0.1 * restart, radius=1.0)
                x0 += np.random.randn(len(x0)) * 0.3
            else:
                # Random perturbation of best
                x0 = best_x + np.random.randn(len(x0)) * 0.2
            
            try:
                res = minimize(
                    smoothed_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-3.0, 3.0)] * (N - 2),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Evaluate raw objective at the result
                raw_val = -res.fun if np.isfinite(res.fun) else 1.0
                actual_score = -raw_objective(res.x)
                
                if actual_score > best_score:
                    best_x = res.x.copy()
                    best_score = actual_score
            except Exception:
                pass
    
    # Phase 3: Final high-precision refinement
    if time.time() - start_time < time_limit - 5.0:
        # Use very high beta for near-exact minimum
        final_beta = 2000.0
        try:
            res_final = minimize(
                smoothed_objective,
                best_x,
                args=(final_beta,),
                method='L-BFGS-B',
                bounds=[(-3.0, 3.0)] * (N - 2),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            actual_score = -raw_objective(res_final.x)
            if actual_score > best_score:
                best_x = res_final.x.copy()
                best_score = actual_score
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish on raw objective
    if time.time() - start_time < time_limit - 2.0:
        try:
            res_nm = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            actual_score = -raw_objective(res_nm.x)
            if actual_score > best_score:
                best_x = res_nm.x.copy()
                best_score = actual_score
        except Exception:
            pass
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
