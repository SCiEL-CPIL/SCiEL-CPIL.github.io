# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds
    safety_margin = 30  # seconds before limit to stop
    
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_points(pts):
        """
        Normalize points to have zero mean and unit standard deviation per dimension?
        Or just zero mean and scale so that max distance is 1?
        Scaling to max distance 1 makes the objective (d_min/1)^2 = d_min^2.
        This is simpler. We want to maximize d_min subject to d_max = 1.
        This is equivalent to maximizing (d_min/d_max)^2.
        """
        center = np.mean(pts, axis=0)
        pts_centered = pts - center
        dists = pdist(pts_centered)
        if len(dists) > 0:
            d_max = np.max(dists)
            if d_max > 1e-12:
                pts_scaled = pts_centered / d_max
            else:
                pts_scaled = pts_centered
        else:
            pts_scaled = pts_centered
        return pts_scaled

    def band_objective(x, beta=0.05, gamma=0.1):
        """
        Smooth objective: normalize distances by soft_max, then penalize
        deviation from the ideal band [1-gamma, 1+gamma].
        This is smoother than the ratio objective near the optimum.
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        # Normalize to prevent scale drift
        p_norm = normalize_points(p)
        
        dists = pdist(p_norm, metric='euclidean')
        
        if np.all(dists == 0):
            return 1e10

        # Soft max for normalization (approximation of d_max)
        # We use a small beta for the soft max to be close to the true max
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_max <= 1e-12:
            return 1e10

        # Normalized distances relative to soft_max
        norm_dists = dists / soft_max
        
        # Penalize distances outside [1-gamma, 1+gamma]
        # We want all distances to be close to soft_max (which is approx d_max)
        # If d_min is also close to d_max, then all dists are close to soft_max.
        # So we penalize dists that are significantly smaller than soft_max.
        # dists are in [0, soft_max]. We want them to be close to soft_max.
        # Let's define deviation from 1.0 (since we divided by soft_max)
        deviations = np.abs(norm_dists - 1.0)
        # We only care about the lower bound really, because norm_dists <= 1 usually 
        # (soft_max >= max(dists) is false, soft_max approximates max, so some might be >1 due to exp approx? 
        # Actually log-sum-exp >= max, so soft_max >= max(dists). 
        # So norm_dists <= 1. 
        # We want to maximize the minimum of norm_dists.
        # So we penalize values significantly less than 1.
        
        # Smooth hinge for lower bound: max(0, (1-gamma) - norm_dists)^2
        lower_violation = np.maximum((1.0 - gamma) - norm_dists, 0.0)
        penalty = np.sum(lower_violation ** 2)
        
        # Also add a small regularization to prevent collapse
        # We already normalized, so spread is fixed. 
        # But if all points collapse, d_max=0, handled above.
        
        return penalty

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        x: flat array of length 42
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        # Normalize
        p_norm = normalize_points(p)
        dists = pdist(p_norm, metric='euclidean')
        
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 1e-12 or soft_max <= 1e-12:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        
        # Normalize for stability
        p_norm = normalize_points(p)
        dists = pdist(p_norm, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    # Initial guesses
    def generate_initial_guesses():
        guesses = []
        
        # 1. Icosahedron + 2 Poles
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        norms_ico = np.linalg.norm(ico_verts, axis=1)
        ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
        points_init = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
        points_init = points_init - np.mean(points_init, axis=0)
        guesses.append(points_init)

        # 2. Fibonacci Sphere
        def fibonacci_sphere(n_points, seed=0):
            rng = np.random.RandomState(seed)
            points = []
            for i in range(n_points):
                z = 1 - 2 * (i + 0.5) / n_points
                r = np.sqrt(1 - z**2)
                theta = 2 * np.pi * i / n_points
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y, z])
            pts = np.array(points)
            pts = pts - np.mean(pts, axis=0)
            return pts
        
        for s in [1, 2, 3, 4, 5]:
            guesses.append(fibonacci_sphere(n, seed=s))

        # 3. Random Points on Sphere
        for s in [10, 20, 30]:
            rng = np.random.RandomState(s)
            pts = rng.randn(n, d)
            norms = np.linalg.norm(pts, axis=1)
            pts = pts / norms[:, np.newaxis]
            pts = pts - np.mean(pts, axis=0)
            guesses.append(pts)
            
        return guesses

    best_points = None
    best_score = -1.0

    initial_guesses = generate_initial_guesses()
    
    # 1. Global Optimization using Differential Evolution
    # We run DE on the smooth objective with a relatively loose beta
    if time.time() - start_time < time_limit - safety_margin:
        try:
            de_deadline = start_time + (time_limit * 0.40)
            
            def de_objective(x):
                if time.time() > de_deadline:
                    return 1e10
                # Use a larger beta for DE to ensure smoothness
                return smooth_objective(x, beta=0.2)
            
            # Bounds: Since we normalize inside, the actual values don't matter as much,
            # but we need to stay within a range where normalization is stable.
            # [-5, 5] is safe.
            bounds = [(-5, 5)] * (n * d)
            
            # Use one of the initial guesses as the starting point for DE?
            # DE generates its own population. We can seed it with the best guess.
            # But standard DE is fine.
            
            res_de = differential_evolution(
                de_objective, 
                bounds=bounds, 
                maxiter=60, 
                popsize=15, 
                tol=1e-6, 
                seed=42,
                init='latinhypercube'
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 2. Local Refinement using L-BFGS-B with Adaptive Beta/Gamma
    # We will take the best point from DE (or initial guesses if DE failed) and polish it.
    # We will also polish the initial guesses themselves.
    
    candidates_to_polish = []
    if best_points is not None:
        candidates_to_polish.append(best_points.copy())
    
    for guess in initial_guesses:
        candidates_to_polish.append(guess.copy())
    
    # Add some random perturbations of the best point found so far
    if best_points is not None:
        np.random.seed(123)
        for _ in range(10):
            perturb = best_points + np.random.randn(n, d) * 0.2
            candidates_to_polish.append(perturb)

    # Sort candidates by their current score to try the best ones first
    # (Optional, but good practice)
    def sort_key(pts):
        return -get_score(pts)
    
    candidates_to_polish.sort(key=sort_key)

    for init_pts in candidates_to_polish:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # Stage 1: Smooth objective with moderate beta
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    # Stage 2: Sharper objective with smaller beta
                    if time.time() - start_time < time_limit - safety_margin:
                        res_local2 = minimize(
                            lambda x: smooth_objective(x, beta=0.01), 
                            opt_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_local2.fun < 1e9:
                            opt_points2 = res_local2.x.reshape((n, d))
                            opt_score2 = get_score(opt_points2)
                            if opt_score2 > best_score:
                                best_points = opt_points2
                                best_score = opt_score2
                                
                                # Stage 3: Band objective for stability
                                if time.time() - start_time < time_limit - safety_margin:
                                    res_local3 = minimize(
                                        lambda x: band_objective(x, beta=0.01, gamma=0.05), 
                                        opt_points2.ravel(), 
                                        method='L-BFGS-B', 
                                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                                    )
                                    if res_local3.fun < 1e9:
                                        opt_points3 = res_local3.x.reshape((n, d))
                                        opt_score3 = get_score(opt_points3)
                                        if opt_score3 > best_score:
                                            best_points = opt_points3
                                            best_score = opt_score3

        except Exception:
            pass

    # 3. Final high-precision polish with exact objective
    if best_points is not None and time.time() - start_time < time_limit - safety_margin:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # 4. Try a few more random restarts if time allows
    if best_points is not None and time.time() - start_time < time_limit - 15:
        for seed in [777, 888, 999, 1000, 1111, 2222, 3333, 4444, 5555]:
            if time.time() - start_time > time_limit - 10:
                break
            try:
                # Random sphere start
                rng = np.random.RandomState(seed)
                rp = rng.randn(n, d)
                rp = rp / np.linalg.norm(rp, axis=1)[:, np.newaxis]
                rp = rp - np.mean(rp, axis=0)
                
                # Quick smooth polish
                res_q = minimize(
                    lambda x: smooth_objective(x, beta=0.03), 
                    rp.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                if res_q.fun < 1e9:
                    qp = res_q.x.reshape((n, d))
                    qs = get_score(qp)
                    if qs > best_score:
                        best_points = qp
                        best_score = qs
                        
                        # Quick exact polish
                        if time.time() - start_time < time_limit - 10:
                            res_q2 = minimize(
                                exact_objective, 
                                qp.ravel(), 
                                method='L-BFGS-B', 
                                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                            )
                            if res_q2.fun < 1e9:
                                qp2 = res_q2.x.reshape((n, d))
                                qs2 = get_score(qp2)
                                if qs2 > best_score:
                                    best_points = qp2
                                    best_score = qs2
            except Exception:
                pass

    # Final validity check
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback to the first initial guess
        return initial_guesses[0]

    return best_points

# EVOLVE-BLOCK-END
