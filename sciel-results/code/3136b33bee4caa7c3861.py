"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import random


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples."""
    n = pts.shape[0]
    min_area = float('inf')
    # Vectorized: compute all cross products
    # For efficiency with 13 points, just loop
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha


def _soft_min_triangle_area(pts, temperature):
    """Soft-min approximation for gradient-based optimization."""
    n = pts.shape[0]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -T * log(sum(exp(-area/T)))
    # For numerical stability
    a = areas - areas.min()
    soft_min = areas.min() - temperature * np.log(np.sum(np.exp(-a / temperature)))
    return soft_min


def _objective_soft(x, temperature):
    """Objective for optimization: minimize -soft_min_area / hull_area."""
    pts = x.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    sm = _soft_min_triangle_area(pts, temperature)
    return -sm / ha


def _regularize(pts, hull_pts_idx):
    """Add small regularization to prevent degenerate configurations."""
    # Not used directly, but we ensure points are well-spread
    pass


def _try_lattice_start(rng):
    """Generate a structured starting point based on a perturbed grid."""
    # Use a 4x4 grid (16 points) and remove 3 to get 13, or use a different structure
    # Better: use a hexagonal-ish packing
    pts = []
    # Try a few structured arrangements
    
    # Arrangement 1: 4x4 grid minus 3 corners
    grid = np.array([[i * 0.333, j * 0.333] for i in range(4) for j in range(4)], dtype=float)
    # Remove 3 points from corners
    remove = [0, 15, 12]  # corners
    keep = [i for i in range(16) if i not in remove]
    pts = grid[keep]
    # Add small random perturbation
    pts = pts + rng.uniform(-0.02, 0.02, size=(13, 2))
    return pts


def _try_hex_start(rng):
    """Hexagonal packing arrangement."""
    # Hexagonal close packing for 13 points
    # Row 0: 4 points
    # Row 1: 3 points (offset)
    # Row 2: 4 points
    # Row 3: 2 points (offset)
    pts = []
    spacing = 0.3
    # Row 0: 4 points
    for i in range(4):
        pts.append([i * spacing, 0.0])
    # Row 1: 3 points, offset by spacing/2
    for i in range(3):
        pts.append([i * spacing + spacing / 2, spacing * 0.866])
    # Row 2: 4 points
    for i in range(4):
        pts.append([i * spacing, 2 * spacing * 0.866])
    # Row 3: 2 points, offset
    for i in range(2):
        pts.append([i * spacing + spacing / 2, 3 * spacing * 0.866])
    pts = np.array(pts)
    pts = pts + rng.uniform(-0.02, 0.02, size=(13, 2))
    return pts


def _try_random_start(rng):
    """Random points in unit square."""
    return np.array([[rng.random(), rng.random()] for _ in range(N)])


def _polish(pts, time_budget):
    """Polish a point configuration using soft-min optimization."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Try multiple temperatures
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget * 0.7:
            break
        
        x0 = best_pts.flatten()
        
        def obj(x):
            return _objective_soft(x, temp)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if result.success or result.fun < obj(x0):
                new_pts = result.x.reshape(-1, 2)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass
        
        if time.time() - start_time > time_budget * 0.8:
            break
    
    # Final direct polish with Nelder-Mead on the actual objective
    if time.time() - start_time < time_budget * 0.9:
        try:
            def direct_obj(x):
                pts_x = x.reshape(-1, 2)
                s = _score(pts_x)
                return -s
            
            result = minimize(direct_obj, best_pts.flatten(), method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            new_pts = result.x.reshape(-1, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score


def _anneal(pts, time_budget):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _score(pts)
    current_pts = pts.copy()
    current_score = best_score
    
    T = 0.05
    T_min = 1e-6
    alpha = 0.85
    steps_per_temp = 20
    
    while T > T_min and time.time() - start_time < time_budget:
        for _ in range(steps_per_temp):
            if time.time() - start_time > time_budget:
                break
            # Perturb
            new_pts = current_pts + np.random.normal(0, T * 0.5, size=(N, 2))
            new_score = _score(new_pts)
            
            if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
                current_pts = new_pts
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        
        T *= alpha
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    rng = random.Random(42)
    np.random.seed(42)
    
    best_pts = None
    best_score = 0.0
    
    # Generate multiple starting configurations
    starts = []
    
    # Structured starts
    starts.append(_try_lattice_start(rng))
    starts.append(_try_hex_start(rng))
    
    # Random starts
    for _ in range(5):
        starts.append(_try_random_start(rng))
    
    # Evaluate all starts
    for s in starts:
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_pts = s.copy()
    
    # Time budget for optimization
    elapsed = time.time() - (deadline - 145)
    remaining = max(10, 145 - elapsed - 5)  # 5s safety margin
    
    # Allocate time to polish the best few starts
    top_starts = sorted(starts, key=lambda p: _score(p), reverse=True)[:4]
    
    # Also include the best found so far
    if best_pts not in top_starts:
        top_starts.append(best_pts)
    
    time_per_start = remaining / len(top_starts)
    
    for i, s in enumerate(top_starts):
        if time.time() > deadline - 10:
            break
        
        # Polish
        polished_pts, polished_score = _polish(s, time_per_start)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
        
        # Anneal if time remains
        if time.time() < deadline - 15:
            anneal_budget = min(5, (deadline - 15 - time.time()) * 0.5)
            if anneal_budget > 1:
                annealed_pts, annealed_score = _anneal(polished_pts, anneal_budget)
                if annealed_score > best_score:
                    best_score = annealed_score
                    best_pts = annealed_pts.copy()
    
    # Final check
    if best_pts is None:
        best_pts = np.array([[rng.random(), rng.random()] for _ in range(N)])
    
    return best_pts
# EVOLVE-BLOCK-END
