# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

DEADLINE = 620.0  # safety margin before 630s kill


def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2


def _spherical_to_cartesian(theta_phi, n=14):
    """Convert (theta, phi) pairs to 3D unit sphere points.
    theta: polar angle [0, pi]
    phi: azimuthal angle [0, 2*pi]
    """
    thetas = theta_phi[0::2]
    phis = theta_phi[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])


def _neg_score_spherical(theta_phi, n=14):
    """Objective: negative of score for minimization."""
    points = _spherical_to_cartesian(theta_phi, n)
    if not np.all(np.isfinite(points)):
        return 1.0  # large penalty
    return -_score(points)


def _random_initial(n=14, seed=None):
    """Generate random initial spherical coordinates."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    thetas = rng.uniform(0.1, np.pi - 0.1, n)
    phis = rng.uniform(0, 2 * np.pi, n)
    theta_phi = np.empty(2 * n)
    theta_phi[0::2] = thetas
    theta_phi[1::2] = phis
    return theta_phi


def _optimize_from(x0, n=14, time_limit=30.0):
    """Run Nelder-Mead from initial point x0."""
    start_time = time.time()
    result = minimize(
        _neg_score_spherical,
        x0,
        args=(n,),
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'maxfev': 5000,
            'xatol': 1e-10,
            'fatol': 1e-10,
        }
    )
    elapsed = time.time() - start_time
    return result.x, result.fun, elapsed


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses multi-start Nelder-Mead optimization on spherical coordinates
    (points constrained to unit sphere) to find good configurations.
    """
    n = 14
    global DEADLINE
    DEADLINE = time.time() + 620.0
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > DEADLINE - 30:
            break
        
        # Generate random initial spherical coordinates
        thetas = rng.uniform(0.1, np.pi - 0.1, n)
        phis = rng.uniform(0, 2 * np.pi, n)
        theta_phi = np.empty(2 * n)
        theta_phi[0::2] = thetas
        theta_phi[1::2] = phis
        
        # Also try some structured initializations
        if i == 0:
            # Uniform distribution on sphere
            z = rng.uniform(-1, 1, n)
            r = np.sqrt(1 - z**2)
            phi = rng.uniform(0, 2*np.pi, n)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            thetas = np.arccos(z)
            phis = phi
            theta_phi = np.empty(2 * n)
            theta_phi[0::2] = thetas
            theta_phi[1::2] = phis
        elif i == 1:
            # Fibonacci sphere
            golden_angle = np.pi * (3 - np.sqrt(5))
            thetas = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
            phis = golden_angle * np.arange(n)
            theta_phi = np.empty(2 * n)
            theta_phi[0::2] = thetas
            theta_phi[1::2] = phis
        elif i == 2:
            # Random with slight bias toward equator
            thetas = rng.uniform(np.pi/4, 3*np.pi/4, n)
            phis = rng.uniform(0, 2 * np.pi, n)
            theta_phi = np.empty(2 * n)
            theta_phi[0::2] = thetas
            theta_phi[1::2] = phis
        
        # Run optimization
        try:
            result_x, result_fun, elapsed = _optimize_from(theta_phi, n, time_limit=30.0)
        except Exception:
            continue
        
        # Evaluate score
        points = _spherical_to_cartesian(result_x, n)
        if not np.all(np.isfinite(points)):
            continue
        
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Check time
        if time.time() > DEADLINE - 60:
            break
    
    # If no valid result, return random points
    if best_points is None:
        rng = np.random.default_rng(0)
        best_points = rng.randn(n, 3)
    
    # Normalize to unit sphere for consistency (optional but helps)
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    best_points = best_points / norms
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = np.random.default_rng(0).randn(n, 3)
    
    return best_points


# EVOLVE-BLOCK-END
