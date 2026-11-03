"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute the maximum radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between any pair / 2
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    min_pair_r = min_pair / 2.0
    
    # Min distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair_r, min_boundary)

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with hexagonal lattice
    best_points = _hexagonal_initial()
    best_score = _compute_score(best_points)
    
    # Simulated annealing
    rng = np.random.default_rng(42)
    current_points = best_points.copy()
    current_score = best_score
    
    T = 0.01  # Initial temperature
    T_min = 1e-8
    cooling = 0.999
    max_iter = 500000
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, T * 0.1, size=2)
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Ensure points stay in unit circle
        norm = np.linalg.norm(new_points[idx])
        if norm > 1.0:
            new_points[idx] = new_points[idx] / norm * 0.999
        
        new_score = _compute_score(new_points)
        
        # Accept or reject
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    # Final polish with scipy
    if time.time() < deadline - 20:
        def objective(x):
            pts = x.reshape(N, 2)
            # Penalty for points outside unit circle
            norms = np.linalg.norm(pts, axis=1)
            boundary_violation = np.maximum(0, norms - 1.0)
            # Min pair distance
            min_pair = np.inf
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_pair:
                        min_pair = d
            return -min(min_pair / 2.0, 1.0 - np.max(norms)) + 10.0 * np.sum(boundary_violation)
        
        result = minimize(objective, best_points.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 1000, 'ftol': 1e-12})
        polished_points = result.x.reshape(N, 2)
        polished_score = _compute_score(polished_points)
        
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    return best_points
# EVOLVE-BLOCK-END
