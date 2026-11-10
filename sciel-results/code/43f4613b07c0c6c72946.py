# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, basinhopping

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a multi-stage optimization:
    1. Differential evolution for global search
    2. Basin hopping with smooth surrogate
    3. Sequential refinement with increasing smoothness
    4. Final exact-objective polish
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better. Normalizes to unit sphere for scale invariance."""
        if len(points) < 2:
            return 0.0
        # Normalize to unit sphere - objective is scale invariant
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        points = points / norms
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def smooth_objective(points, beta=50.0):
        """
        Smooth surrogate: maximize soft_min(dists) / max(dists).
        Normalizes to unit sphere for scale invariance.
        """
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        points = points / norms
        
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_max):
            return 0.0
        min_d = dists.min()
        shifted = -beta * (dists - min_d)
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = min_d - log_sum / beta
        
        if not np.isfinite(soft_min) or soft_min <= 0:
            return 0.0
        
        ratio = soft_min / d_max
        if not np.isfinite(ratio):
            return 0.0
        return ratio * ratio

    def neg_smooth(x, beta=50.0):
        """Negative smooth objective for minimization."""
        points = x.reshape(n, d)
        return -smooth_objective(points, beta)

    def neg_objective(x):
        """Negative exact objective for minimization."""
        points = x.reshape(n, d)
        return -objective(points)

    def sphere_project(x):
        """Project points onto unit sphere."""
        points = x.reshape(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return (points / norms).flatten()

    # Initialization strategies
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def symmetric_start(n_points):
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    def static_solution():
        pts = np.array([
            [ 0.000000,  0.000000,  1.000000],
            [ 0.000000, -1.000000,  0.000000],
            [ 0.000000,  1.000000,  0.000000],
            [ 1.000000,  0.000000,  0.000000],
            [-1.000000,  0.000000,  0.000000],
            [ 0.000000,  0.000000, -1.000000],
            [ 0.577350,  0.577350,  0.577350],
            [ 0.577350, -0.577350, -0.577350],
            [-0.577350,  0.577350, -0.577350],
            [-0.577350, -0.577350,  0.577350],
            [ 0.816497,  0.408248, -0.408248],
            [-0.816497,  0.408248,  0.408248],
            [ 0.408248, -0.816497,  0.408248],
            [-0.408248, -0.816497, -0.408248],
        ])
        return pts

    def parametric_points(params):
        """
        Generate 14 points from a low-dimensional parameterization.
        params = [eps1, eps2, z_amp, phase, rot_x, rot_y]
        - eps1: amplitude of cos(2*theta) radial perturbation
        - eps2: amplitude of cos(3*theta) radial perturbation  
        - z_amp: out-of-plane displacement amplitude
        - phase: phase offset for radial perturbations
        - rot_x, rot_y: rotation angles to break symmetry
        """
        eps1 = params[0]
        eps2 = params[1]
        z_amp = params[2]
        phase = params[3]
        rot_x = params[4]
        rot_y = params[5]
        
        theta = 2 * np.pi * np.arange(n) / n + phase
        r = 1.0 + eps1 * np.cos(2 * theta) + eps2 * np.cos(3 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z_amp * np.sin(2 * theta)
        
        pts = np.column_stack([x, y, z])
        
        # Apply rotation
        cx, sx = np.cos(rot_x), np.sin(rot_x)
        cy, sy = np.cos(rot_y), np.sin(rot_y)
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        
        pts = pts @ Ry.T @ Rx.T
        return pts

    def neg_parametric(params):
        """Negative objective for parametric optimization."""
        pts = parametric_points(params)
        return -objective(pts)
    
    # Phase 0: Sphere-constrained optimization using spherical coordinates
    if not check_deadline():
        try:
            # Use spherical coordinates: each point defined by (theta, phi)
            # x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
            def spherical_to_cartesian(params):
                """Convert spherical params to Cartesian on unit sphere."""
                thetas = params[:n]
                phis = params[n:2*n]
                x = np.sin(thetas) * np.cos(phis)
                y = np.sin(thetas) * np.sin(phis)
                z = np.cos(thetas)
                return np.column_stack([x, y, z])
            
            def neg_spherical(params):
                """Negative objective in spherical coordinates."""
                pts = spherical_to_cartesian(params)
                return -objective(pts)
            
            # Initial spherical params: evenly distributed
            theta0 = np.linspace(0, np.pi, n, endpoint=False) + 0.1
            phi0 = np.linspace(0, 2*np.pi, n, endpoint=False)
            init_params = np.concatenate([theta0, phi0])
            
            # Optimize in spherical space
            bounds_sph = [(0.01, np.pi-0.01)] * n + [(0, 2*np.pi)] * n
            
            res_sph = differential_evolution(
                neg_spherical,
                bounds_sph,
                maxiter=150,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=True
            )
            
            if not check_deadline():
                sph_points = spherical_to_cartesian(res_sph.x)
                sph_score = objective(sph_points)
                
                if sph_score > best_score:
                    best_score = sph_score
                    best_points = sph_points.copy()
                    
                # Refine in Cartesian space from spherical result
                x0 = sph_points.flatten()
                for beta in [100.0, 200.0, 500.0]:
                    if check_deadline():
                        break
                    res = minimize(
                        lambda x, b=beta: neg_smooth(x, b),
                        x0,
                        method='Powell',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    x0 = res.x
                
                if not check_deadline():
                    res_final = minimize(
                        neg_objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    opt_points = res_final.x.reshape(n, d)
                    opt_score = objective(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        
        except Exception:
            pass
    
    # Phase 1: Differential evolution for global optimization (on unit sphere)
    if not check_deadline():
        try:
            bounds = [(-1.0, 1.0)] * (n * d)
            de_result = differential_evolution(
                neg_smooth,
                bounds,
                args=(30.0,),
                maxiter=120,
                popsize=20,
                seed=42,
                tol=1e-6,
                polish=True
            )
            
            if de_result.success or not check_deadline():
                de_points = de_result.x.reshape(n, d)
                # Normalize to sphere
                norms = np.linalg.norm(de_points, axis=1, keepdims=True)
                norms[norms < 1e-15] = 1.0
                de_points = de_points / norms
                de_score = objective(de_points)
                
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Multi-start basin hopping + refinement
    init_strategies = []
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("symmetric", symmetric_start(n)))
    init_strategies.append(("static", static_solution()))
    
    # Add parametric initializations with different parameters
    param_inits = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.05, 0.1, 0.3, 0.1, 0.05],
        [0.05, 0.1, 0.15, 0.5, -0.1, 0.1],
        [0.15, 0.05, 0.05, 0.7, 0.15, -0.1],
    ]
    for i, p in enumerate(param_inits):
        init_strategies.append((f"parametric_{i}", parametric_points(p)))
    
    for seed in range(12):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    # Add best DE result if found
    if best_points is not None:
        init_strategies.append(("de_best", best_points.copy()))
    
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        # Normalize initial points to sphere
        norms = np.linalg.norm(init_points, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        init_points = init_points / norms
        
        x0 = init_points.flatten()
        
        try:
            # Basin hopping with smooth surrogate
            result = basinhopping(
                lambda x: neg_smooth(x, beta=30.0),
                x0,
                minimizer_kwargs={
                    'method': 'Powell',
                    'options': {
                        'maxiter': 500,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                },
                niter=25,
                stepsize=0.6,
                T=80.0,
                seed=42
            )
            
            if check_deadline():
                break
            
            # Sequential refinement with adaptive beta annealing
            x_current = result.x
            
            for beta in [50.0, 100.0, 200.0, 300.0, 500.0, 800.0, 1200.0]:
                if check_deadline():
                    break
                
                res = minimize(
                    lambda x, b=beta: neg_smooth(x, b),
                    x_current,
                    method='Powell',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                x_current = res.x
            
            if check_deadline():
                break
            
            # High-precision SLSQP polish with tight tolerances
            res_slsqp = minimize(
                lambda x: neg_smooth(x, beta=1200.0),
                x_current,
                method='SLSQP',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if check_deadline():
                break
            
            # Final exact-objective Nelder-Mead polish with very tight tolerance
            res_final = minimize(
                neg_objective,
                res_slsqp.x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            
            if check_deadline():
                break
            
            optimized_points = res_final.x.reshape(n, d)
            score = objective(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # If no valid result, use Fibonacci sphere as fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    # Final validation
    if not np.isfinite(best_score) or best_score < 0:
        best_points = fibonacci_sphere(n)
    
    # Final high-precision polish on the best result found
    if not check_deadline():
        try:
            # Normalize to sphere before final polish
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            best_points = best_points / norms
            
            x_best = best_points.flatten()
            
            # Additional SLSQP refinement with very tight tolerances
            res_polish = minimize(
                lambda x: neg_smooth(x, beta=1500.0),
                x_best,
                method='SLSQP',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if not check_deadline():
                # Final Nelder-Mead with extreme precision
                res_final2 = minimize(
                    neg_objective,
                    res_polish.x,
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16}
                )
                
                if not check_deadline():
                    polished_points = res_final2.x.reshape(n, d)
                    polished_score = objective(polished_points)
                    
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = polished_points.copy()
                        
        except Exception:
            pass
    
    # Ensure final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
