"""Combined program: model-and-optimize with log-variance objective and multi-restart."""
import numpy as np
import scipy.optimize
import time
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 110.0  # Safety margin before 120s limit

# EVOLVE-BLOCK-START

def _pairwise_distances(points):
    """Calculate all pairwise distances between points."""
    d = cdist(points, points)
    mask = ~np.eye(N, dtype=bool)
    return d[mask]

def _true_score(points):
    """Calculate the actual score: (max/min)^2."""
    d = _pairwise_distances(points)
    if len(d) == 0:
        return np.inf
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _objective(params):
    """
    Objective function to minimize: variance of log pairwise distances.
    This encourages all pairwise distances to be as similar as possible,
    which minimizes the max/min ratio.
    """
    points = params.reshape((N, D))
    # Center the points to remove translational invariance
    points = points - np.mean(points, axis=0)
    
    d = _pairwise_distances(points)
    
    # Avoid log(0) and penalize close points
    if np.any(d < 1e-12):
        return 1e10
        
    log_d = np.log(d)
    
    # Variance of log distances
    var = np.var(log_d)
    
    return var

def _objective_ratio(params):
    """
    Objective function to minimize: squared ratio of max to min pairwise distance.
    params: flattened array of shape (N*D,)
    Returns: score value (lower is better)
    """
    points = params.reshape((N, D))
    # Center the points to remove translational invariance
    points = points - np.mean(points, axis=0)
    
    d = _pairwise_distances(points)
    
    if np.any(d < 1e-10):
        return 1e10
        
    ratio = d.max() / d.min()
    return ratio ** 2

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_random_points():
    """Generate random points with some structure."""
    rng = np.random.default_rng()
    pts = rng.randn(N, D)
    return pts

def _generate_initial_guess(idx):
    """Generate an initial guess with jitter."""
    if idx == 0:
        # Fibonacci sphere
        pts = _generate_fib_sphere_points()
    else:
        # Random Gaussian
        rng = np.random.default_rng(seed=42 + idx)
        pts = rng.randn(N, D)
    
    # Add some randomness to break symmetry
    rng = np.random.default_rng(seed=100 + idx)
    pts += rng.normal(0, 0.1, size=(N, D))
    
    # Normalize to unit radius for sphere-based starts
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-10] = 1
    pts /= norms[:, np.newaxis]
    
    return pts.flatten()

def _optimize_local(x0, method='L-BFGS-B', obj_func=None, maxiter=500):
    """Run local optimization from x0."""
    if obj_func is None:
        obj_func = _objective
        
    try:
        if method == 'L-BFGS-B':
            res = scipy.optimize.minimize(
                obj_func, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-15}
            )
        elif method == 'Powell':
            res = scipy.optimize.minimize(
                obj_func, 
                x0, 
                method='Powell', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12}
            )
        else:
            res = scipy.optimize.minimize(
                obj_func, 
                x0, 
                method=method, 
                options={'maxiter': maxiter}
            )
        return res.x, res.fun
    except Exception as e:
        return None, np.inf

def construct():
    start_time = time.time()
    
    best_score = np.inf
    best_points = None
    
    # Phase 1: Multi-restart with L-BFGS-B using log-variance objective
    n_restarts_l = 50
    for i in range(n_restarts_l):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        x0 = _generate_initial_guess(i)
        
        opt_x, opt_val = _optimize_local(x0, method='L-BFGS-B', obj_func=_objective, maxiter=300)
        
        if opt_x is not None:
            points = opt_x.reshape((N, D))
            points = points - np.mean(points, axis=0)
            
            d = _pairwise_distances(points)
            if d.min() < 1e-6:
                continue
                
            score = _true_score(points)
            
            if score < best_score:
                best_score = score
                best_points = points
    
    # Phase 2: Refine best points with Powell using ratio objective
    if best_points is not None and time.time() - start_time < DEADLINE - 10:
        x0 = best_points.flatten()
        opt_x, opt_val = _optimize_local(x0, method='Powell', obj_func=_objective_ratio, maxiter=1000)
        
        if opt_x is not None:
            points = opt_x.reshape((N, D))
            points = points - np.mean(points, axis=0)
            
            d = _pairwise_distances(points)
            if d.min() > 1e-6:
                score = _true_score(points)
                if score < best_score:
                    best_score = score
                    best_points = points
    
    # Phase 3: Additional random restarts with Powell if time permits
    if time.time() - start_time < DEADLINE - 20:
        n_restarts_p = 10
        for i in range(n_restarts_p):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            x0 = _generate_initial_guess(100 + i)
            
            opt_x, opt_val = _optimize_local(x0, method='Powell', obj_func=_objective_ratio, maxiter=1000)
            
            if opt_x is not None:
                points = opt_x.reshape((N, D))
                points = points - np.mean(points, axis=0)
                
                d = _pairwise_distances(points)
                if d.min() < 1e-6:
                    continue
                    
                score = _true_score(points)
                
                if score < best_score:
                    best_score = score
                    best_points = points
    
    # If no valid point was found, return a fallback
    if best_points is None:
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points

# EVOLVE-BLOCK-END
