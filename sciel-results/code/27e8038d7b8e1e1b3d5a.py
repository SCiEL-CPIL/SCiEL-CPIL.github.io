"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

_DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _score(points):
    """Return (max_pairwise / min_pairwise)^2.  Lower is better."""
    d = pdist(points)
    d_min = d.min()
    d_max = d.max()
    if d_min < 1e-15:
        return 1e30
    return (d_max / d_min) ** 2


def _obj(points):
    """Objective for scipy.optimize: the ratio (not squared) – smoother landscape."""
    d = pdist(points)
    d_min = d.min()
    if d_min < 1e-12:
        return 1e20
    return d.max() / d_min


def _triangular_lattice(n):
    """Generate n points on a triangular (hexagonal) lattice, centred."""
    # For 16 points, use a hexagonal arrangement: center + rings
    # Ring 0: 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points (need only 9 more)
    pts = [np.array([0.0, 0.0])]
    # Ring 1
    for k in range(6):
        angle = k * np.pi / 3.0
        pts.append(np.array([np.cos(angle), np.sin(angle)]))
    # Ring 2
    for k in range(12):
        angle = k * np.pi / 6.0
        pts.append(2.0 * np.array([np.cos(angle), np.sin(angle)]))
    # Take first n
    pts = pts[:n]
    arr = np.array(pts)
    # Centroid
    arr -= arr.mean(axis=0)
    return arr


def _circular_two_ring(n):
    """Two concentric rings: 1 center + 6 inner + 9 outer (for n=16)."""
    pts = [np.array([0.0, 0.0])]
    # Inner ring: 6 points at radius 1
    for k in range(6):
        angle = k * np.pi / 3.0
        pts.append(np.array([np.cos(angle), np.sin(angle)]))
    # Outer ring: 9 points at radius 2
    for k in range(9):
        angle = k * 2.0 * np.pi / 9.0 + np.pi / 9.0  # offset to avoid alignment
        pts.append(2.0 * np.array([np.cos(angle), np.sin(angle)]))
    pts = pts[:n]
    arr = np.array(pts)
    arr -= arr.mean(axis=0)
    return arr


def _grid_seed(n):
    """Simple square grid seed (fallback)."""
    side = int(np.ceil(n ** 0.5))
    grid = np.array([(i, j) for i in range(side) for j in range(side)][:n], dtype=float)
    grid -= grid.mean(axis=0)
    return grid


def _local_optimize(points, time_budget):
    """Run Powell local optimization with time guard."""
    start = time.time()
    best = points.copy()
    best_val = _obj(best)
    
    x0 = best.ravel()
    result = minimize(_obj, x0, method='Powell',
                      options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10})
    
    elapsed = time.time() - start
    if result.fun < best_val:
        best = result.x.reshape(-1, 2)
        best_val = result.fun
    
    # If we have time, do a second refinement pass
    if time.time() - start < time_budget:
        result2 = minimize(_obj, best.ravel(), method='Powell',
                           options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12})
        if result2.fun < best_val:
            best = result2.x.reshape(-1, 2)
            best_val = result2.fun
    
    return best, _score(best)


def construct():
    t0 = time.time()
    
    # Generate multiple diverse seeds
    seeds = [
        _triangular_lattice(N),
        _circular_two_ring(N),
        _grid_seed(N),
    ]
    
    # Add a few randomised perturbations of the triangular lattice
    rng = np.random.RandomState(42)
    for _ in range(3):
        seed = _triangular_lattice(N) + 0.1 * rng.randn(N, 2)
        seeds.append(seed)
    
    best_points = seeds[0]
    best_score = _score(best_points)
    
    time_per_seed = (_DEADLINE - (time.time() - t0) - 5.0) / len(seeds)
    time_per_seed = max(time_per_seed, 5.0)  # minimum 5s per seed
    
    for seed in seeds:
        elapsed = time.time() - t0
        remaining = _DEADLINE - 10.0 - elapsed  # 10s safety margin
        if remaining < 5.0:
            break
        
        t_budget = min(time_per_seed, remaining)
        pts, sc = _local_optimize(seed, t_budget)
        if sc < best_score:
            best_score = sc
            best_points = pts
    
    # Final check: ensure we return a valid array
    return best_points
# EVOLVE-BLOCK-END
