import numpy as np
import scipy.optimize as so
import time
import itertools

N = 14
D = 3

def _pairwise_distances_squared(X):
    """Compute squared pairwise distances for points X (N x D)."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    return dist2

def _smooth_objective(X_flat):
    """
    Objective: log(max_dist) - log(min_dist)
    This is smoother than (max/min)^2 for optimization.
    """
    X = X_flat.reshape(N, D)
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    
    if len(d2) == 0:
        return 1e10
        
    d_min_sq = np.min(d2)
    d_max_sq = np.max(d2)
    
    if d_min_sq < 1e-12 or d_max_sq < 1e-12:
        return 1e10
        
    # log(sqrt(d_max_sq)) - log(sqrt(d_min_sq)) = 0.5 * (log(d_max_sq) - log(d_min_sq))
    # Minimizing this is equivalent to minimizing the ratio.
    return 0.5 * (np.log(d_max_sq) - np.log(d_min_sq))

def _raw_score(X):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    if len(d2) == 0:
        return 1e10
    d_min_sq = np.min(d2)
    d_max_sq = np.max(d2)
    if d_min_sq < 1e-12:
        return 1e10
    return d_max_sq / d_min_sq

def _generate_fibonacci_sphere():
    """Generate points on a sphere using Fibonacci lattice."""
    N_pts = N
    golden_angle = np.pi * (3. - np.sqrt(5.))
    X = np.zeros((N_pts, 3))
    for i in range(N_pts):
        z = 1. - (i / (N_pts - 1.)) * 2.
        radius = np.sqrt(1. - z**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        X[i] = [x, y, z]
    return X

def _generate_lattice():
    """Generate points from a cubic lattice."""
    # 14 points in a 3x3x3 cube (27 points)
    grid = list(itertools.product(range(3), repeat=3))
    # Select 14 points that are somewhat spread out
    # Simple selection: first 14
    X = np.array(grid, dtype=float)
    X -= np.mean(X, axis=0)
    return X

def _generate_random():
    """Generate random points in a cube."""
    X = np.random.uniform(-1, 1, (N, D))
    X -= np.mean(X, axis=0)
    return X

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 1e10
    best_X = None
    
    # Initial guesses
    inits = [
        _generate_fibonacci_sphere(),
        _generate_lattice(),
        _generate_random(),
        _generate_random(),
        _generate_random(),
        _generate_random()
    ]
    
    for init_idx, X_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        X_flat = X_init.flatten()
        
        try:
            # Use Powell method which is robust for non-smooth or less smooth functions
            # We use the smooth objective to guide the search
            res = so.minimize(
                _smooth_objective,
                X_flat,
                method='Powell',
                options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            X_candidate = res.x.reshape(N, D)
            score = _raw_score(X_candidate)
            
            if score < best_score:
                best_score = score
                best_X = X_candidate
                
        except Exception:
            continue

    if best_X is None:
        # Fallback
        return _generate_fibonacci_sphere()
        
    # Final polish with Nelder-Mead on the raw objective?
    # Or just return the best found.
    # Let's try a quick polish with L-BFGS-B on the smooth objective
    if time.time() < deadline - 5:
        try:
            res2 = so.minimize(
                _smooth_objective,
                best_X.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            X_candidate2 = res2.x.reshape(N, D)
            score2 = _raw_score(X_candidate2)
            if score2 < best_score:
                best_score = score2
                best_X = X_candidate2
        except Exception:
            pass

    return best_X
