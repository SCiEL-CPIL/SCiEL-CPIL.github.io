"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _score(points):
    """Compute the radius score for a set of points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside unit circle
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms >= 1.0):
        return 0.0
    # Compute minimum pairwise distance
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Compute minimum distance to boundary
    min_boundary = 1.0 - np.max(norms)
    r = min(min_pair_dist / 2.0, min_boundary)
    return r

def _random_init():
    """Generate a random feasible initial configuration."""
    points = np.zeros((N, 2))
    for i in range(N):
        for _ in range(10000):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x*x + y*y < 0.81:  # radius 0.9 for safety
                points[i] = [x, y]
                break
        else:
            # Fallback: place on a spiral
            angle = i * 2.399963229728653  # golden angle
            r = 0.9 * np.sqrt(i / N)
            points[i] = [r * np.cos(angle), r * np.sin(angle)]
    return points

def _ring_init():
    """Generate an initial configuration using concentric rings."""
    points = []
    # Center
    points.append([0.0, 0.0])
    
    # Try different ring configurations
    best_points = None
    best_score = -1
    
    for n_rings in range(1, 10):
        pts = [[0.0, 0.0]]
        remaining = N - 1
        # Distribute points among rings
        for ring in range(1, n_rings + 1):
            if remaining <= 0:
                break
            # Number of points on this ring
            n_on_ring = min(remaining, max(1, int(2 * np.pi * ring * 0.15 / (2 * np.pi / 36))))
            n_on_ring = min(n_on_ring, remaining)
            if n_on_ring <= 0:
                continue
            radius = ring * 0.15
            if radius >= 0.95:
                break
            for j in range(n_on_ring):
                angle = 2 * np.pi * j / n_on_ring + np.pi / n_on_ring
                pts.append([radius * np.cos(angle), radius * np.sin(angle)])
            remaining -= n_on_ring
        
        if len(pts) == N:
            pts = np.array(pts)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
    
    if best_points is not None:
        return best_points
    return _random_init()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    # Start with ring initialization
    points = _ring_init()
    best_points = points.copy()
    best_score = _score(points)
    
    # Local optimization using gradient ascent on the score
    # We maximize r = min(min_pair_dist/2, min_boundary)
    
    def _objective(x):
        """Negative score for minimization."""
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    def _objective_and_grad(x):
        """Negative score and its gradient approximation."""
        pts = x.reshape(-1, 2)
        score = _score(pts)
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (-_score(x_plus.reshape(-1, 2)) + _score(x_minus.reshape(-1, 2))) / (2 * eps)
        
        return -score, grad
    
    # Use Nelder-Mead for robustness (no gradient needed)
    x0 = points.flatten()
    
    # Multiple restarts with local optimization
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x0 = points.flatten()
        else:
            # Perturb the best solution
            x0 = best_points.flatten().copy()
            noise = np.random.normal(0, 0.01, x0.shape)
            x0 += noise
        
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            pts = result.x.reshape(-1, 2)
            s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                
        except Exception:
            continue
        
        # Check deadline
        if time.time() > deadline - 10:
            break
    
    # Final refinement with smaller steps if time permits
    if time.time() < deadline - 15:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'initial_simplex': 1e-6}
            )
            pts = result.x.reshape(-1, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
