"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(points):
    """Compute the minimum radius for a set of points."""
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    n = len(points)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    return min(min_boundary, min_pair / 2.0)


def _seed_hexagonal():
    """Generate a hexagonal lattice seed."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def _seed_random():
    """Generate a random seed within the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def _optimize_from_seed(seed, max_iter=1000, tol=1e-12):
    """Optimize from a seed using gradient-free optimization."""
    x0 = seed.flatten()
    
    def neg_score(x):
        pts = x.reshape(-1, 2)
        # Check if any point is outside
        if np.any(np.linalg.norm(pts, axis=1) > 1.0):
            return 1.0
        return -_score(pts)
    
    # Use Nelder-Mead for robustness
    res = minimize(neg_score, x0, method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
    return res.x.reshape(-1, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Try multiple seeds
    seeds = []
    
    # Hexagonal seed
    try:
        hex_seed = _seed_hexagonal()
        seeds.append(hex_seed)
    except Exception:
        pass
    
    # Random seeds
    for _ in range(5):
        try:
            rand_seed = _seed_random()
            seeds.append(rand_seed)
        except Exception:
            pass
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        try:
            optimized = _optimize_from_seed(seed, max_iter=500)
            score = _score(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            continue
    
    if best_points is None:
        # Fallback to hexagonal
        best_points = _seed_hexagonal()
    
    return best_points
# EVOLVE-BLOCK-END
