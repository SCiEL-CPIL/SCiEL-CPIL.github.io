# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, basinhopping

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a multi-stage optimization:
    1. Sphere-constrained optimization with spherical coordinates
    2. Global optimization with differential evolution
    3. Local refinement with Nelder-Mead
    4. Multiple restarts with different strategies
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 15.0  # seconds
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
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
    
    def neg_objective(points):
        """Negative objective for minimization."""
        return -objective(points)
    
    def sphere_objective(params):
        """
        Objective for sphere-constrained optimization.
        params: array of shape (n, 3) with spherical coordinates (theta, phi, r)
        theta: azimuthal angle [0, 2*pi)
        phi: polar angle [0, pi]
        r: radius [0.5, 1.5]
        """
        theta = params[:, 0]
        phi = params[:, 1]
        r = params[:, 2]
        
        # Convert to Cartesian
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points = np.column_stack([x, y, z])
        
        return neg_objective(points)
    
    def flat_objective(x):
        """Flat objective for unconstrained optimization."""
        points = x.reshape(n, d)
        return neg_objective(points)
    
    best_points = None
    best_score = -1.0
    
    def update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if not np.all(np.isfinite(points)):
            return
        score = objective(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy 1: Fibonacci sphere initialization
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
    
    # Strategy 2: Icosahedron-based initialization
    def icosahedron_start(n_points):
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        if n_points == 12:
            return icosahedron
        
        # Add 2 more points for 14
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    # Strategy 3: Random sphere
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    # Strategy 4: Random Gaussian
    def random_gaussian(n_points, seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_points, 3))
    
    # Generate initial points
    inits = []
    inits.append(("fib", fibonacci_sphere(n)))
    inits.append(("icosa", icosahedron_start(n)))
    
    for seed in range(15):
        inits.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    for seed in range(15):
        inits.append((f"rand_gauss_{seed}", random_gaussian(n, seed + 100)))
    
    # Phase 1: Sphere-constrained optimization using spherical coordinates
    # Parameters: (theta, phi, r) for each point
    # theta in [0, 2*pi], phi in [0, pi], r in [0.5, 1.5]
    
    def sphere_to_cartesian(params):
        """Convert spherical params to Cartesian."""
        theta = params[:, 0]
        phi = params[:, 1]
        r = params[:, 2]
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.column_stack([x, y, z])
    
    def sphere_flat_objective(x):
        """Flat objective for spherical parameterization."""
        params = x.reshape(n, 3)
        points = sphere_to_cartesian(params)
        return neg_objective(points)
    
    # Bounds for spherical parameters
    sphere_bounds = [
        (0, 2 * np.pi),  # theta
        (0, np.pi),      # phi
        (0.5, 1.5)       # r
    ]
    sphere_bounds_flat = [b for b in sphere_bounds for _ in range(n)]
    
    # Phase 2: Unconstrained optimization
    
    for name, init_points in inits:
        if check_deadline():
            break
        
        # Try sphere-constrained optimization
        try:
            # Convert initial points to spherical coordinates
            r = np.linalg.norm(init_points, axis=1)
            theta = np.arctan2(init_points[:, 1], init_points[:, 0])
            phi = np.arccos(init_points[:, 2] / np.where(r == 0, 1, r))
            
            # Ensure valid ranges
            theta = np.where(theta < 0, theta + 2 * np.pi, theta)
            phi = np.clip(phi, 0, np.pi)
            r = np.clip(r, 0.5, 1.5)
            
            params_init = np.column_stack([theta, phi, r]).flatten()
            
            # Use differential evolution for global optimization on sphere
            result_de = differential_evolution(
                sphere_flat_objective,
                bounds=sphere_bounds_flat,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            
            if result_de.success and np.all(np.isfinite(result_de.x)):
                params_opt = result_de.x.reshape(n, 3)
                points_opt = sphere_to_cartesian(params_opt)
                update_best(points_opt)
            
            # Also try local refinement from DE result
            if np.all(np.isfinite(result_de.x)):
                result_local = minimize(
                    sphere_flat_objective,
                    result_de.x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if np.all(np.isfinite(result_local.x)):
                    params_opt = result_local.x.reshape(n, 3)
                    points_opt = sphere_to_cartesian(params_opt)
                    update_best(points_opt)
                    
        except Exception:
            pass
        
        if check_deadline():
            break
        
        # Try unconstrained optimization with basin hopping
        try:
            x0 = init_points.flatten()
            
            result_bh = basinhopping(
                flat_objective,
                x0,
                minimizer_kwargs={
                    'method': 'Nelder-Mead',
                    'options': {
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                },
                niter=30,
                stepsize=0.5,
                T=50.0,
                seed=42
            )
            
            if np.all(np.isfinite(result_bh.x)):
                points_opt = result_bh.x.reshape(n, d)
                update_best(points_opt)
                
        except Exception:
            pass
        
        if check_deadline():
            break
        
        # Try direct Nelder-Mead from initial
        try:
            x0 = init_points.flatten()
            result_nm = minimize(
                flat_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if np.all(np.isfinite(result_nm.x)):
                points_opt = result_nm.x.reshape(n, d)
                update_best(points_opt)
                
        except Exception:
            pass
    
    # Phase 3: If we have a good candidate, do more intensive local refinement
    if best_points is not None and not check_deadline():
        # Multiple Nelder-Mead refinements with different initial perturbations
        for perturb_scale in [0.01, 0.05, 0.1, 0.2]:
            if check_deadline():
                break
            
            rng = np.random.default_rng(12345)
            for trial in range(5):
                if check_deadline():
                    break
                
                perturbed = best_points + rng.standard_normal((n, d)) * perturb_scale
                x0 = perturbed.flatten()
                
                try:
                    result = minimize(
                        flat_objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    
                    if np.all(np.isfinite(result.x)):
                        points_opt = result.x.reshape(n, d)
                        update_best(points_opt)
                        
                except Exception:
                    pass
    
    # Phase 4: Final intensive optimization on the best candidate
    if best_points is not None and not check_deadline():
        # Try basin hopping with more iterations on the best
        try:
            x0 = best_points.flatten()
            
            result_bh = basinhopping(
                flat_objective,
                x0,
                minimizer_kwargs={
                    'method': 'Nelder-Mead',
                    'options': {
                        'maxiter': 5000,
                        'xatol': 1e-12,
                        'fatol': 1e-14
                    }
                },
                niter=50,
                stepsize=0.3,
                T=30.0,
                seed=999
            )
            
            if np.all(np.isfinite(result_bh.x)):
                points_opt = result_bh.x.reshape(n, d)
                update_best(points_opt)
                
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
