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
    
    def smoothed_objective(x, beta=100.0):
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
    def initial_guess(seed_offset=0):
        x = np.zeros(2 * (N - 2))
        # Place points on a unit circle, skipping first two
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + seed_offset
        for i in range(N - 2):
            idx = i + 2
            x[2 * i] = np.cos(angles[idx])
            x[2 * i + 1] = np.sin(angles[idx])
        return x
    
    # Try multiple random restarts with different initializations
    best_score = -np.inf
    best_x = None
    
    # Time limit management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Phase 1: Differential evolution for global search
    bounds = [(-3.0, 3.0) for _ in range(2 * (N - 2))]
    
    try:
        result_de = differential_evolution(
            raw_objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=12,
            tol=1e-6,
            init='sobol'
        )
        best_x = result_de.x
        best_score = -result_de.fun
    except Exception:
        pass
    
    if time.time() - start_time > time_limit:
        return get_points(best_x)
    
    # Phase 2: Local refinement with smoothed objective
    # Try several different beta values for smoothing
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        # Multiple restarts
        for restart in range(3):
            if time.time() - start_time > time_limit:
                break
            
            x0 = initial_guess(restart * 0.5)
            # Add some randomness
            x0 += np.random.randn(len(x0)) * 0.5
            
            try:
                res = minimize(
                    smoothed_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-3.0, 3.0)] * (N - 2),
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                if res.fun < -best_score:
                    best_x = res.x
                    best_score = -res.fun
            except Exception:
                pass
    
    # Phase 3: Final refinement with Nelder-Mead on raw objective
    if time.time() - start_time < time_limit:
        try:
            res_local = minimize(
                raw_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_local.fun < -best_score:
                best_x = res_local.x
                best_score = -res_local.fun
        except Exception:
            pass
    
    return get_points(best_x)

# EVOLVE-BLOCK-END
