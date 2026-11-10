# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses multi-stage softness refinement with symmetric initialization.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    def get_score(pts):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_scale(pts):
        """Normalize so that d_max = 1 to enforce scale invariance."""
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return pts
        d_max = np.max(dists)
        if d_max < 1e-12:
            return pts
        return pts / d_max

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        Scale-invariant: normalizes internally.
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        # Enforce scale invariance by normalizing
        dists_raw = pdist(p, metric='euclidean')
        d_max_raw = np.max(dists_raw)
        if d_max_raw < 1e-12:
            return 1e10
        p_norm = p / d_max_raw
        dists = pdist(p_norm, metric='euclidean')
        
        # Soft min
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        if max_a_neg > 500:
            return 1e10
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        # Soft max
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        if max_a_pos > 500:
            return 1e10
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 1e-9 or soft_max <= 1e-9:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 1e-9 or d_max <= 1e-9:
            return 1e10
        return (d_max / d_min) ** 2

    def make_icosahedron_plus_poles():
        """Create 12 icosahedron vertices + 2 poles, centered and scaled."""
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        norms_ico = np.linalg.norm(ico_verts, axis=1)
        ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
        
        poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        points_init = np.vstack([ico_verts_norm, poles])
        
        # Center
        points_init = points_init - np.mean(points_init, axis=0)
        return points_init

    def make_fibonacci_sphere(n_points):
        """Deterministic Fibonacci sphere."""
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius = np.sqrt(max(0, 1 - y**2))
            theta = golden_angle * i
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points[i] = [x, y, z]
        points = points - np.mean(points, axis=0)
        return points

    def make_random_symmetric(seed_val):
        """Generate a random configuration with some symmetry."""
        rng = np.random.RandomState(seed_val)
        pts = rng.randn(n, d) * 0.5
        # Add slight icosahedron bias
        ico = make_icosahedron_plus_poles()
        pts = 0.7 * ico + 0.3 * pts
        pts = pts - np.mean(pts, axis=0)
        return pts

    # 1. Multiple initializations
    initializations = [
        make_icosahedron_plus_poles(),
        make_fibonacci_sphere(n),
        make_random_symmetric(1),
        make_random_symmetric(42),
        make_random_symmetric(123),
    ]
    
    best_points = initializations[0].copy()
    best_score = get_score(best_points)
    
    for init in initializations[1:]:
        s = get_score(init)
        if s > best_score:
            best_points = init.copy()
            best_score = s

    # 2. Brief global search with DE (reduced budget)
    if time.time() < deadline - 180.0:
        try:
            de_deadline = time.time() + 80.0
            
            def de_objective(x):
                if time.time() > de_deadline:
                    return 1e10
                return smooth_objective(x, beta=0.15)
            
            bounds = [(-3, 3)] * (n * d)
            
            res_de = differential_evolution(
                de_objective, 
                bounds=bounds, 
                maxiter=30, 
                popsize=15, 
                tol=1e-3, 
                seed=42,
                init='latinhypercube',
                polish=False
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 3. Multi-stage local refinement with decreasing beta
    # Start from best point and several perturbations
    candidates_to_polish = [best_points.copy()]
    
    # Add perturbations
    rng = np.random.RandomState(42)
    for scale in [0.05, 0.1, 0.2, 0.3]:
        perturb = best_points + rng.randn(n, d) * scale
        candidates_to_polish.append(perturb)
    
    # Also add the initial symmetric points
    candidates_to_polish.append(make_icosahedron_plus_poles())
    candidates_to_polish.append(make_fibonacci_sphere(n))

    # Multi-stage softness refinement
    beta_stages = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for init_pts in candidates_to_polish:
        if time.time() > deadline - 60.0:
            break
            
        current_pts = init_pts.copy()
        
        for beta in beta_stages:
            if time.time() > deadline - 30.0:
                break
            
            try:
                res_local = minimize(
                    lambda x, b=beta: smooth_objective(x, beta=b), 
                    current_pts.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                if res_local.fun < 1e9:
                    opt_points = res_local.x.reshape((n, d))
                    opt_score = get_score(opt_points)
                    
                    if opt_score > best_score:
                        best_points = opt_points
                        best_score = opt_score
                        current_pts = opt_points.copy()
                    else:
                        # Keep current_pts if it's still good
                        current_pts = opt_points.copy()
                        
            except Exception:
                break

    # 4. Final exact polish with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # 5. One more round of exact polish from multiple starting points
    if time.time() < deadline - 5.0:
        try:
            for start_pt in [best_points, make_icosahedron_plus_poles(), make_fibonacci_sphere(n)]:
                if time.time() > deadline - 2.0:
                    break
                res_extra = minimize(
                    exact_objective, 
                    start_pt.ravel(), 
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_extra.fun < 1e9:
                    extra_points = res_extra.x.reshape((n, d))
                    extra_score = get_score(extra_points)
                    if extra_score > best_score:
                        best_points = extra_points
                        best_score = extra_score
        except Exception:
            pass

    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = make_icosahedron_plus_poles()

    return best_points

# EVOLVE-BLOCK-END
