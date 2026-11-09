# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

_N = 14


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


def _spherical_to_cartesian(theta_phi, n=_N):
    """Convert (theta, phi) pairs to 3D unit sphere points."""
    thetas = theta_phi[0::2]
    phis = theta_phi[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])


def _neg_score_spherical(theta_phi, n=_N):
    """Objective: negative of score for minimization."""
    points = _spherical_to_cartesian(theta_phi, n)
    if not np.all(np.isfinite(points)):
        return 1.0
    return -_score(points)


def _soft_min_max_score(theta_phi, n=_N, sigma=0.05):
    """Soft surrogate using log-sum-exp for smooth optimization."""
    points = _spherical_to_cartesian(theta_phi, n)
    if not np.all(np.isfinite(points)):
        return 1.0
    dists = pdist(points, metric='euclidean')
    if len(dists) == 0:
        return 1.0
    
    d_scaled = dists / sigma
    d_scaled = np.clip(d_scaled, -500, 500)
    
    # Soft min
    max_neg = np.max(-d_scaled)
    soft_min = -sigma * (max_neg + np.log(np.sum(np.exp(-d_scaled - max_neg))))
    
    # Soft max
    max_d = np.max(d_scaled)
    soft_max = sigma * (max_d + np.log(np.sum(np.exp(d_scaled - max_d))))
    
    if soft_max <= 0:
        return 1.0
    
    ratio = soft_min / soft_max
    return -(ratio ** 2)


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points as initial guess."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    thetas = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    phis = golden_angle * np.arange(n)
    theta_phi = np.empty(2 * n)
    theta_phi[0::2] = thetas
    theta_phi[1::2] = phis
    return theta_phi


def _random_spherical(rng, n, bias='uniform'):
    """Generate random spherical coordinates with optional bias."""
    if bias == 'equator':
        thetas = rng.uniform(np.pi/4, 3*np.pi/4, n)
    elif bias == 'polar':
        thetas = np.concatenate([
            rng.uniform(0.1, np.pi/3, n//2),
            rng.uniform(2*np.pi/3, np.pi-0.1, n - n//2)
        ])
    else:
        thetas = rng.uniform(0.05, np.pi - 0.05, n)
    phis = rng.uniform(0, 2 * np.pi, n)
    theta_phi = np.empty(2 * n)
    theta_phi[0::2] = thetas
    theta_phi[1::2] = phis
    return theta_phi


def _coordinate_descent_polish(theta_phi, n=_N, max_iterations=500, time_limit=20.0):
    """Polish using coordinate descent on spherical coordinates with adaptive step size."""
    start = time.time()
    x = theta_phi.copy()
    current_score = _score(_spherical_to_cartesian(x, n))
    
    base_deltas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for it in range(max_iterations):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            # Optimize theta_i
            best_theta = x[2*i]
            best_theta_score = current_score
            for delta in base_deltas:
                for sign in [1, -1]:
                    new_x = x.copy()
                    new_x[2*i] = best_theta + sign * delta
                    new_score = _score(_spherical_to_cartesian(new_x, n))
                    if new_score > best_theta_score:
                        best_theta_score = new_score
                        best_theta = new_x[2*i]
                        improved = True
            x[2*i] = best_theta
            
            # Optimize phi_i
            best_phi = x[2*i+1]
            best_phi_score = current_score
            for delta in base_deltas:
                for sign in [1, -1]:
                    new_x = x.copy()
                    new_x[2*i+1] = best_phi + sign * delta
                    new_score = _score(_spherical_to_cartesian(new_x, n))
                    if new_score > best_phi_score:
                        best_phi_score = new_score
                        best_phi = new_x[2*i+1]
                        improved = True
            x[2*i+1] = best_phi
            
            current_score = best_theta_score
        
        if not improved:
            break
    
    return x


def _nelder_mead_polish(theta_phi, n=_N, time_limit=10.0):
    """Polish using Nelder-Mead with tight tolerances."""
    start = time.time()
    try:
        result = minimize(
            _neg_score_spherical,
            theta_phi,
            args=(n,),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 5000,
                'xatol': 1e-12,
                'fatol': 1e-12,
            }
        )
        return result.x
    except Exception:
        return theta_phi


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    
    Uses multi-phase optimization:
    1. Multi-start with soft surrogate
    2. Nelder-Mead on raw score
    3. Coordinate descent polish
    4. Final high-precision Nelder-Mead polish
    """
    n = _N
    deadline = time.time() + 580.0  # safety margin before 630s kill
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # Fibonacci sphere
    initial_guesses.append(_fibonacci_sphere(n))
    
    # Uniform random
    initial_guesses.append(_random_spherical(rng, n, bias='uniform'))
    
    # Equator-biased
    initial_guesses.append(_random_spherical(rng, n, bias='equator'))
    
    # Polar-biased
    initial_guesses.append(_random_spherical(rng, n, bias='polar'))
    
    # More random starts
    for _ in range(6):
        initial_guesses.append(_random_spherical(rng, n, bias='uniform'))
    
    # Process each initial guess
    for idx, theta_phi in enumerate(initial_guesses):
        if time.time() > deadline - 40:
            break
        
        # Phase 1: Soft surrogate optimization
        try:
            result_soft = minimize(
                _soft_min_max_score,
                theta_phi,
                args=(n, 0.05),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                }
            )
        except Exception:
            result_soft = type('obj', (), {'x': theta_phi})()
        
        if time.time() > deadline - 30:
            break
        
        # Phase 2: Raw score Nelder-Mead from soft result
        try:
            result_raw = minimize(
                _neg_score_spherical,
                result_soft.x,
                args=(n,),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'maxfev': 3000,
                    'xatol': 1e-11,
                    'fatol': 1e-11,
                }
            )
        except Exception:
            result_raw = result_soft
        
        if time.time() > deadline - 25:
            break
        
        # Phase 3: Coordinate descent polish
        polished_x = _coordinate_descent_polish(result_raw.x, n, max_iterations=300, time_limit=15.0)
        
        points = _spherical_to_cartesian(polished_x, n)
        if np.all(np.isfinite(points)):
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # Phase 4: Final high-precision Nelder-Mead polish
        if time.time() > deadline - 15:
            break
        
        final_x = _nelder_mead_polish(polished_x, n, time_limit=10.0)
        final_points = _spherical_to_cartesian(final_x, n)
        if np.all(np.isfinite(final_points)):
            final_score = _score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
    
    # If no valid result, return Fibonacci sphere
    if best_points is None:
        theta_phi = _fibonacci_sphere(n)
        best_points = _spherical_to_cartesian(theta_phi, n)
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = rng.randn(n, 3)
    
    # Ensure shape is correct
    if best_points.shape != (n, 3):
        best_points = rng.randn(n, 3)
    
    return best_points


# EVOLVE-BLOCK-END
