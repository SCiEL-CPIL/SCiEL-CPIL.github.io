# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 625  # Leave 5s margin from 630s limit
    
    # Precompute indices for all 286 triangles
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    i_idx, j_idx, k_idx = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]

    def calculate_score(points):
        # points: (13, 2)
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Vectorized triangle area calculation
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0
            
        return A_min / A_hull

    def neg_score(points):
        s = calculate_score(points)
        return -s

    def soft_min_score(points, temperature=0.01):
        """Smooth surrogate using soft-min of triangle areas"""
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Soft-min using log-sum-exp trick
        log_areas = np.log(np.maximum(areas, 1e-15))
        max_log = np.max(log_areas)
        soft_min = max_log + np.log(np.sum(np.exp(log_areas - max_log))) - np.log(len(areas))
        
        # Convert back from log space
        A_min_soft = np.exp(soft_min)
        
        if A_min_soft <= 1e-10:
            return 0.0
            
        return A_min_soft / A_hull

    def neg_soft_score(points):
        s = soft_min_score(points)
        return -s

    def normalize_points(points):
        """Center and normalize points to unit radius"""
        centered = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 1e-10:
            return centered / max_dist
        return centered

    best_score = 0.0
    best_points = None

    def update_best(points, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(reg_poly)
    
    rng = np.random.default_rng(42)
    
    # 2. Jittered Regular Polygon (15 variations)
    for _ in range(15):
        jitter = rng.normal(0, 0.15, (n, 2))
        guess = reg_poly + jitter
        initial_guesses.append(guess)
        
    # 3. Random points (15 variations)
    for _ in range(15):
        guess = rng.random((n, 2))
        initial_guesses.append(guess)

    # Phase 1: Quick evaluation of initial guesses
    for guess in initial_guesses:
        if time.time() - start_time > time_limit - 30:
            break
        norm_guess = normalize_points(guess)
        score = calculate_score(norm_guess)
        update_best(norm_guess, score)

    # Phase 2: Differential Evolution for global optimization
    if time.time() - start_time < time_limit - 60:
        try:
            # Define bounds for normalized space
            bounds = [(-1.5, 1.5)] * (2 * n)
            
            def de_objective(x):
                pts = x.reshape(n, 2)
                return -calculate_score(pts)
            
            de_result = scipy.optimize.differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-6,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            
            de_points = de_result.x.reshape(n, 2)
            de_score = calculate_score(de_points)
            update_best(de_points, de_score)
        except Exception:
            pass

    # Phase 3: Nelder-Mead refinement from multiple starting points
    if best_points is not None and time.time() - start_time < time_limit - 40:
        # Start from best found so far
        start_points = [best_points.copy()]
        
        # Also try from some initial guesses
        for guess in initial_guesses[:5]:
            start_points.append(normalize_points(guess))
        
        for sp in start_points:
            if time.time() - start_time > time_limit - 20:
                break
            
            try:
                result = scipy.optimize.minimize(
                    neg_score, 
                    sp, 
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = result.x.copy()
            except Exception:
                pass

    # Phase 4: Powell refinement with soft-min surrogate for smoother landscape
    if best_points is not None and time.time() - start_time < time_limit - 30:
        try:
            result_powell = scipy.optimize.minimize(
                neg_soft_score, 
                best_points, 
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if result_powell.fun < -best_score:
                best_score = -result_powell.fun
                best_points = result_powell.x.copy()
        except Exception:
            pass

    # Phase 5: Final high-precision polish with exact objective
    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            result_final = scipy.optimize.minimize(
                neg_score, 
                best_points, 
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result_final.fun < -best_score:
                best_score = -result_final.fun
                best_points = result_final.x.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = initial_guesses[0]
        
    return best_points

# EVOLVE-BLOCK-END
