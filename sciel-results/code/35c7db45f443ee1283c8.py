# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580s limit, leaving 50s margin for safety

    n = 14
    d = 3

    def get_score(points):
        """Compute the metric (d_min/d_max)^2"""
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x, beta):
        """
        Smoothed objective using log-sum-exp approximation.
        We want to maximize (d_min/d_max)^2.
        Equivalent to maximizing log(d_min) - log(d_max).
        Using: -logsumexp(-beta * log(dists)) for log(d_min) approximation?
        Actually: log(min(d)) ≈ -1/beta * logsumexp(-beta * log(d))
        log(max(d)) ≈ 1/beta * logsumexp(beta * log(d))
        
        So we want to maximize: -1/beta * logsumexp(-beta * log(d)) - 1/beta * logsumexp(beta * log(d))
        To minimize, we negate this.
        """
        points = x.reshape(n, d)
        if not np.all(np.isfinite(points)):
            return 1e10
        dists = pdist(points)
        if np.any(dists <= 0):
            return 1e10
        
        # Clip dists to avoid log(0)
        dists = np.clip(dists, 1e-10, None)
        log_dists = np.log(dists)
        
        # log(d_min) approx: -1/beta * logsumexp(-beta * log_dists)
        # log(d_max) approx:  1/beta * logsumexp( beta * log_dists)
        
        # To avoid overflow in exp, shift by max
        def logsumexp(v):
            m = np.max(v)
            if not np.isfinite(m):
                return 0.0
            return m + np.log(np.sum(np.exp(v - m)))
        
        min_term = -1.0 / beta * logsumexp(-beta * log_dists)
        max_term = 1.0 / beta * logsumexp(beta * log_dists)
        
        # We want to maximize (min_term - max_term)
        # So minimize -(min_term - max_term)
        return -(min_term - max_term)

    # --- Initialization Strategies ---
    
    def icosahedron_init():
        phi = (1 + np.sqrt(5)) / 2
        ico_vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
        
        # Add 2 points. Try poles first.
        extra_points = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
        points = np.vstack([ico_vertices, extra_points])
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def fibonacci_sphere_init():
        points = np.zeros((n, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i] = [x, y, z]
        return points

    def random_init(seed):
        np.random.seed(seed)
        points = np.random.randn(n, d)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return points

    # List of initializers
    initializers = [
        icosahedron_init,
        fibonacci_sphere_init,
        lambda: random_init(42),
        lambda: random_init(123),
        lambda: random_init(999),
    ]

    best_score = -1.0
    best_points = None

    # Multi-stage optimization with annealing
    # Start with low beta (smooth) and increase to high beta (sharp)
    betas = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for init_idx, init_func in enumerate(initializers):
        if time.time() > deadline - 30:
            break
            
        points = init_func()
        x0 = points.flatten()
        
        # Run annealing stages
        current_points = points.copy()
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            # Optimization call
            try:
                res = minimize(
                    objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < 1e9: # Valid result
                    opt_points = res.x.reshape(n, d)
                    # Normalize to unit sphere to maintain scale invariance
                    opt_points = opt_points / np.linalg.norm(opt_points, axis=1, keepdims=True)
                    
                    score = get_score(opt_points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = opt_points.copy()
                        
                    # Use this as start for next stage
                    x0 = opt_points.flatten()
                    
            except Exception:
                pass

    # Final polish with Nelder-Mead on the best point found, using exact objective
    if best_points is not None and time.time() < deadline - 5:
        def exact_objective(x):
            points = x.reshape(n, d)
            return -get_score(points)
            
        try:
            res_final = minimize(
                exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_final.fun < 0: # i.e. score > 0
                final_points = res_final.x.reshape(n, d)
                final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
                final_score = get_score(final_points)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = icosahedron_init()

    return best_points


# EVOLVE-BLOCK-END
