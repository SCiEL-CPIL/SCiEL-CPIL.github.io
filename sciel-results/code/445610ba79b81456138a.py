# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, basinhopping

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    # Normalize to unit sphere to enforce Tammes problem structure
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    pts_norm = points / norms
    
    dists = pdist(pts_norm)
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
    
    Uses a multi-start local optimization approach starting from
    structured initial configurations (including Cuboctahedron + poles)
    and optimizing with a smooth surrogate objective (soft-min) on the
    unit sphere, followed by high-precision polish.
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
        """Returns (d_min/d_max)^2, higher is better. Points are normalized."""
        if len(points) < 2:
            return 0.0
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts_norm = points / norms
        
        dists = pdist(pts_norm)
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
        Smooth surrogate: maximize soft_min(dists) / max(dists)
        Points are normalized to unit sphere.
        """
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts_norm = points / norms
        
        dists = pdist(pts_norm)
        if len(dists) == 0:
            return 0.0
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_max):
            return 0.0
        
        # soft-min of distances
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

    # Initialization strategies
    init_strategies = []
    
    # Strategy 1: Fibonacci sphere
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
    
    # Strategy 2: Icosahedron-based symmetric start
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
    
    # Strategy 3: Random uniform on sphere
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    # Strategy 4: Cuboctahedron + Poles (High Symmetry Start)
    def cubocta_start():
        # 12 vertices of a cuboctahedron: permutations of (1, 1, 0)
        # Signs are independent
        verts = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                verts.append([x, y, 0])
                verts.append([x, 0, y])
                verts.append([0, x, y])
        # This generates 12 unique points?
        # (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0) -> 4
        # (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1) -> 4
        # (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1) -> 4
        # Total 12.
        verts = np.array(verts)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts /= norms
        
        # Add 2 poles
        p13 = np.array([0.0, 0.0, 1.0])
        p14 = np.array([0.0, 0.0, -1.0])
        
        return np.vstack([verts, [p13, p14]])
    
    # Strategy 5: Pre-computed static solution (refined)
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
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms
    
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("symmetric", symmetric_start(n)))
    init_strategies.append(("cubocta", cubocta_start()))
    init_strategies.append(("static", static_solution()))
    
    for seed in range(15):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    for seed in range(10):
        rng = np.random.default_rng(seed + 100)
        pts = rng.standard_normal((n, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        init_strategies.append((f"rand_gauss_{seed}", pts))
    
    # Main optimization loop
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        try:
            # Phase 1: Basin hopping with smooth surrogate
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
                stepsize=1.0,
                T=100.0,
                accept_test=None,
                seed=42
            )
            
            if check_deadline():
                break
            
            # Phase 2: Refine with higher beta smooth objective
            x_refined = result.x
            result2 = minimize(
                lambda x: neg_smooth(x, beta=100.0),
                x_refined,
                method='Powell',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if check_deadline():
                break
            
            # Phase 3: Polish with SLSQP on smooth objective
            x_slsqp = result2.x
            result3 = minimize(
                lambda x: neg_smooth(x, beta=200.0),
                x_slsqp,
                method='SLSQP',
                options={'maxiter': 800, 'ftol': 1e-14}
            )
            
            if check_deadline():
                break
            
            # Phase 4: Polish with L-BFGS-B
            x_lbfgs = result3.x
            result4 = minimize(
                lambda x: neg_smooth(x, beta=300.0),
                x_lbfgs,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if check_deadline():
                break
            
            # Phase 5: Final polish with exact objective using Nelder-Mead
            x_final = result4.x
            result5 = minimize(
                neg_objective,
                x_final,
                method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            if check_deadline():
                break
            
            # Phase 6: Additional exact polish
            x_extra = result5.x
            result6 = minimize(
                lambda x: neg_smooth(x, beta=500.0),
                x_extra,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            
            if check_deadline():
                break
            
            result7 = minimize(
                neg_objective,
                result6.x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            
            if check_deadline():
                break
            
            # Evaluate both candidates
            optimized_points_a = result5.x.reshape(n, d)
            optimized_points_b = result7.x.reshape(n, d)
            
            # Normalize final points for consistency
            norms_a = np.linalg.norm(optimized_points_a, axis=1, keepdims=True)
            norms_a[norms_a == 0] = 1.0
            optimized_points_a /= norms_a
            
            norms_b = np.linalg.norm(optimized_points_b, axis=1, keepdims=True)
            norms_b[norms_b == 0] = 1.0
            optimized_points_b /= norms_b
            
            score_a = objective(optimized_points_a)
            score_b = objective(optimized_points_b)
            
            if score_b > score_a:
                optimized_points = optimized_points_b
                score = score_b
            else:
                optimized_points = optimized_points_a
                score = score_a
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # If no valid result, use Cuboctahedron start as fallback
    if best_points is None:
        best_points = cubocta_start()
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = cubocta_start()
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = cubocta_start()
        
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms

    return best_points

# EVOLVE-BLOCK-END
