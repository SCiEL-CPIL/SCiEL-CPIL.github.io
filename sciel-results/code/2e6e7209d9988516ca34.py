"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import math

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area_vec(pts):
    """Vectorized minimum triangle area computation."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    
    # Precompute all index combinations
    indices = list(itertools.combinations(range(N), 3))
    idx_array = np.array(indices, dtype=np.int32)
    p = pts[idx_array]  # (286, 3, 2)
    
    A = p[:, 0, :]
    B = p[:, 1, :]
    C = p[:, 2, :]
    
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return float(np.min(areas))

def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_vec(pts)
    return min_tri / hull_a

def _negative_score(flat_pts):
    """Negative score for minimization."""
    pts = flat_pts.reshape(N, 2)
    s = _score(pts)
    return -s

def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0, 1) for the given base."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        j = i
        while j > 0:
            f /= base
            j, r = divmod(j, base)
            x += f * r
        result.append(x)
    return result

def _halton_points(n, seed=0):
    """Generate n well-spread 2D points using Halton sequences."""
    rng = np.random.RandomState(seed)
    # Use two different bases for x and y
    xs = _halton_sequence(n, 2)
    ys = _halton_sequence(n, 3)
    # Add small jitter to avoid exact degeneracies
    jitter = rng.uniform(-0.01, 0.01, size=(n, 2))
    pts = np.column_stack([xs, ys]) + jitter
    return pts

def _jittered_grid(n, seed=42, spacing=0.1):
    """Generate a jittered grid of points."""
    rng = np.random.RandomState(seed)
    # Create a grid and select n points
    side = int(math.ceil(math.sqrt(n)))
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            x = (i + 0.5) / side + rng.uniform(-spacing/2, spacing/2)
            y = (j + 0.5) / side + rng.uniform(-spacing/2, spacing/2)
            points.append([x, y])
    return np.array(points[:n])

def _polish_nelder_mead(pts_init, deadline, maxiter=500):
    """Polish using Nelder-Mead optimization."""
    if time.time() > deadline:
        return pts_init, _score(pts_init)
    
    flat_init = pts_init.flatten()
    
    try:
        result = minimize(
            _negative_score,
            flat_init,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        polished_pts = result.x.reshape(N, 2)
        polished_score = _score(polished_pts)
        return polished_pts, polished_score
    except Exception:
        return pts_init, _score(pts_init)

def _multi_start_local(deadline, n_starts=8):
    """
    Multi-start local optimization:
    1. Generate diverse initial points (Halton, jittered grid, random)
    2. Polish each with Nelder-Mead
    3. Return the best result
    """
    best_pts = None
    best_score = 0.0
    
    # Generate diverse initial points
    initial_sets = []
    
    # Halton-based
    for seed in range(3):
        initial_sets.append(_halton_points(N, seed=seed))
    
    # Jittered grid
    for seed in [42, 123]:
        initial_sets.append(_jittered_grid(N, seed=seed))
    
    # Random points in [0, 1]^2
    rng = np.random.RandomState(999)
    for seed in [42, 123]:
        initial_sets.append(rng.uniform(0, 1, size=(N, 2)))
    
    # Process each initial set
    for i, init_pts in enumerate(initial_sets):
        if time.time() > deadline - 5.0:  # Leave 5s buffer
            break
        
        try:
            polished_pts, polished_score = _polish_nelder_mead(init_pts, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
        except Exception:
            continue
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 145s
    deadline = start_time + 145.0
    
    # Multi-start local optimization
    best_pts, best_score = _multi_start_local(deadline, n_starts=8)
    
    # If we still have time, do a second round with different starts
    if best_pts is not None and time.time() < deadline - 20.0:
        # Generate more diverse starts
        rng = np.random.RandomState(777)
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            init_pts = rng.uniform(-0.5, 0.5, size=(N, 2))
            try:
                polished_pts, polished_score = _polish_nelder_mead(init_pts, deadline)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished_pts
            except Exception:
                continue
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: return a reasonable grid
    return _jittered_grid(N)

# EVOLVE-BLOCK-END
