"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _pairwise_dists(X):
    """Compute all pairwise distances for points X of shape (n, d)."""
    # X: (n, d)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    return dists

def _objective_smooth(X_flat, tau=100.0):
    """
    Smooth surrogate for (max_dist / min_dist)^2.
    Uses softmax/softmin to make it differentiable.
    """
    X = X_flat.reshape(N, D)
    # Compute pairwise squared distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    # Set diagonal to 0 (will be excluded)
    sq_dists = np.where(np.eye(N, dtype=bool), 0.0, sq_dists)
    
    # Use log-sum-exp trick for soft max/min of distances
    # We want soft max of dists = soft max of sq_dists (since sqrt is monotone)
    # Actually, let's work with distances directly for clarity
    dists = np.sqrt(sq_dists + 1e-12)
    dists = np.where(np.eye(N, dtype=bool), 1e-10, dists)
    
    # Soft max: exp(tau * max) approximation
    # Use log-sum-exp for stability
    d_max_log = np.log(np.sum(np.exp(tau * dists - tau * np.max(dists)), axis=None)) / tau + np.max(dists)
    d_min_log = -np.log(np.sum(np.exp(-tau * dists + tau * np.min(dists)), axis=None)) / tau + np.min(dists)
    
    ratio = d_max_log / d_min_log
    return ratio**2

def _objective_exact(X_flat):
    """Exact (non-smooth) objective."""
    X = X_flat.reshape(N, D)
    dists = pdist(X)
    if np.min(dists) < 1e-10:
        return 1e10
    return (np.max(dists) / np.min(dists))**2

def _fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 0.0
        radius = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _random_start(seed):
    """Generate a random start on a sphere."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2*np.pi, N)
    phi = np.arccos(rng.uniform(-1, 1, N))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _polyhedron_start():
    """Start from cuboctahedron-like arrangement plus 2 extra points."""
    # Cuboctahedron has 12 vertices
    # Let's use icosahedron (12 vertices) + 2 poles? No, icosahedron doesn't have poles.
    # Let's use: 2 poles + 6 equatorial + 6 at mid-latitude
    pts = []
    # Poles
    pts.append([0, 0, 1])
    pts.append([0, 0, -1])
    # 6 points on equator
    for i in range(6):
        angle = 2*np.pi*i/6
        pts.append([np.cos(angle), np.sin(angle), 0])
    # 6 points at latitude 45 deg
    lat = np.pi/4
    r = np.cos(lat)
    z = np.sin(lat)
    for i in range(6):
        angle = 2*np.pi*i/6 + np.pi/6  # offset by 30 deg
        pts.append([r*np.cos(angle), r*np.sin(angle), z])
    return np.array(pts)

def _optimize_from_start(X0, deadline, best_result, best_score):
    """Run optimization from a single start. Returns (result, score)."""
    X0 = X0.reshape(-1)
    best_local = X0
    best_local_score = _objective_exact(X0)
    
    # Try multiple tau values for the smooth surrogate
    for tau in [50.0, 200.0, 500.0, 1000.0]:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                lambda x: _objective_smooth(x, tau=tau),
                X0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < best_local_score:
                X0 = res.x
                best_local = res.x
                best_local_score = _objective_exact(res.x)
        except Exception:
            pass
    
    # Final polish with SLSQP on the exact objective (using subgradient approximation)
    if time.time() > deadline - 5:
        pass
    else:
        try:
            res = minimize(
                lambda x: _objective_exact(x),
                best_local,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_local_score:
                best_local = res.x
                best_local_score = res.fun
        except Exception:
            pass
    
    if best_local_score < best_score:
        return best_local, best_local_score
    return None, best_score

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Generate diverse initial configurations
    starts = []
    
    # 1. Fibonacci sphere
    starts.append(_fibonacci_sphere(N))
    
    # 2. Polyhedron-based start
    starts.append(_polyhedron_start())
    
    # 3. Several random starts
    for seed in range(10):
        starts.append(_random_start(seed))
    
    # 4. Grid-based start (from original)
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    starts.append(np.array(grid, dtype=float))
    
    best_result = None
    best_score = np.inf
    
    for i, X0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        result, score = _optimize_from_start(X0, deadline, best_result, best_score)
        if result is not None:
            best_result = result
            best_score = score
    
    if best_result is None:
        # Fallback to Fibonacci sphere
        return _fibonacci_sphere(N)
    
    return best_result.reshape(N, D)
# EVOLVE-BLOCK-END
