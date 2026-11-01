"""Pack 50 equal circles in a unit circle using constrained optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def _score(centers):
    """Compute the packing radius (the score)."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    r_boundary = np.min(dist_boundary)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = np.min(dist) / 2.0
    return min(r_boundary, r_pair)

def _negative_score(centers_flat):
    """Negative score for minimization."""
    centers = centers_flat.reshape(N, 2)
    return -_score(centers)

def _hex_init(scale=1.0):
    """Hexagonal lattice initialization."""
    spacing = 0.24 * scale
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y < 0.85:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Sort by distance from center, take closest N
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:N]
    return points[idx]

def _ring_init():
    """Ring-based initialization."""
    centers = []
    # Center
    centers.append([0.0, 0.0])
    # Rings
    ring_radii = [0.2, 0.4, 0.6, 0.8]
    ring_counts = [6, 12, 18, 13]
    for r, count in zip(ring_radii, ring_counts):
        for k in range(count):
            angle = 2 * np.pi * k / count + (r * 0.5)
            centers.append([r * np.cos(angle), r * np.sin(angle)])
    centers = np.array(centers[:N])
    return centers

def _random_init():
    """Random initialization inside unit circle."""
    centers = []
    for _ in range(N):
        while True:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y < 0.81:
                centers.append([x, y])
                break
    return np.array(centers)

def _optimize(centers):
    """Run SLSQP optimization on the given initial centers."""
    x0 = centers.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Constraints: keep points inside unit circle (with margin)
    def constraint_i(i):
        def c(x):
            return 1.0 - 0.02 - x[2*i]**2 - x[2*i+1]**2
        c.name = f'c{i}'
        return c
    
    constraints = [{'type': 'ineq', 'fun': constraint_i(i)} for i in range(N)]
    
    result = minimize(
        _negative_score,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
    )
    
    if result.success or result.fun < _negative_score(x0):
        return result.x.reshape(N, 2), -result.fun
    return centers, _score(centers)

def _local_refine(centers, max_iter=50):
    """Simple local refinement: iteratively push apart overlapping circles."""
    centers = centers.copy()
    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(N):
            for j in range(i+1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 0.24 and dist > 1e-10:
                    # Push apart
                    push = d / dist * (0.24 - dist) * 0.5
                    centers[i] += push
                    centers[j] -= push
                    improved = True
            # Push toward center if too close to boundary
            r = np.linalg.norm(centers[i])
            if r > 0.98:
                centers[i] *= 0.98 / r
                improved = True
        if not improved:
            break
    return centers

def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations
    inits = []
    
    # Hexagonal
    for scale in [1.0, 0.95, 1.05, 0.9]:
        h = _hex_init(scale)
        if h is not None:
            inits.append(h)
    
    # Ring
    inits.append(_ring_init())
    
    # Random
    np.random.seed(42)
    for _ in range(3):
        inits.append(_random_init())
    
    for init_centers in inits:
        if time.time() > DEADLINE - 5:
            break
        
        # Local refine first
        refined = _local_refine(init_centers, max_iter=30)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_centers = refined.copy()
        
        # Full optimization
        opt_centers, opt_score = _optimize(refined)
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers.copy()
    
    # Final local refine on best
    if best_centers is not None:
        final = _local_refine(best_centers, max_iter=100)
        s = _score(final)
        if s > best_score:
            best_score = s
            best_centers = final
    
    if best_centers is None:
        # Fallback
        best_centers = _hex_init(1.0)
        if best_centers is None:
            best_centers = _random_init()
    
    return best_centers
