"""Stronger packing algorithm for 50 equal circles in unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_score(points):
    """Compute the maximum radius for given points (vectorized)."""
    n = len(points)
    if n < 2:
        return 0.0
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair, min_boundary)

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing, then optimize."""
    # Try multiple spacings to get a good initial configuration
    best_pts = None
    best_score = -1
    
    for spacing in np.linspace(0.25, 0.4, 20):
        # Generate hexagonal lattice
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        points = []
        for row in range(-10, 11):
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(-10, 11):
                x = col * dx + offset
                y = row * dy
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        if len(points) < N:
            continue
        
        points = np.array(points)
        # Select N points closest to center
        dists_center = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists_center)[:N]
        pts = points[idx]
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        # Fallback: random points
        rng = np.random.default_rng(42)
        best_pts = rng.uniform(-0.8, 0.8, size=(N, 2))
    
    return best_pts

def _anneal(points, time_budget, seed=42):
    """Simulated annealing with vectorized score computation."""
    rng = np.random.default_rng(seed)
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    start = time.time()
    T = 0.05
    T_min = 1e-9
    cooling = 0.9995
    
    while time.time() - start < time_budget:
        # Move a random point
        idx = rng.integers(0, N)
        # Perturbation size scales with temperature
        scale = T * 0.5
        delta = rng.normal(0, scale, size=2)
        new_pt = current[idx] + delta
        
        # Keep inside unit circle
        norm = np.linalg.norm(new_pt)
        if norm > 1.0:
            new_pt = new_pt / norm * 0.999
        
        current[idx] = new_pt
        new_score = _compute_score(current)
        
        # Accept/reject
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-15)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            current[idx] -= delta  # revert
        
        T *= cooling
        if T < T_min:
            break
    
    return best, best_score

def _polish(points):
    """Polish using scipy minimize with a smooth objective."""
    def objective(x):
        pts = x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        boundary_violation = np.sum(np.maximum(0, norms - 1.0))
        
        # Smooth min of pair distances
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        # Use soft min: -log(sum(exp(-d/T))) * T
        T_soft = 0.01
        soft_min = -T_soft * np.log(np.sum(np.exp(-dists / T_soft)))
        
        return -soft_min / 2.0 - (1.0 - np.max(norms)) + 100.0 * boundary_violation
    
    try:
        result = minimize(objective, points.flatten(), method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-14})
        polished = result.x.reshape(N, 2)
        return polished
    except:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Initial hexagonal packing
    points = _hexagonal_initial()
    score = _compute_score(points)
    
    # Multiple annealing restarts with different seeds
    best_points = points.copy()
    best_score = score
    
    time_per_anneal = 30.0
    max_restarts = 3
    
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        elapsed = time.time() - start_time
        remaining = deadline - 20 - elapsed
        if remaining < 5:
            break
        
        t_budget = min(time_per_anneal, remaining / (max_restarts - restart))
        
        # Anneal from current best
        annealed, a_score = _anneal(best_points, t_budget, seed=100 + restart)
        if a_score > best_score:
            best_score = a_score
            best_points = annealed.copy()
        
        # Polish
        if time.time() < deadline - 15:
            polished = _polish(best_points)
            p_score = _compute_score(polished)
            if p_score > best_score:
                best_score = p_score
                best_points = polished
    
    # Final polish
    if time.time() < deadline - 5:
        polished = _polish(best_points)
        p_score = _compute_score(polished)
        if p_score > best_score:
            best_points = polished
            best_score = p_score
    
    return best_points
# EVOLVE-BLOCK-END
