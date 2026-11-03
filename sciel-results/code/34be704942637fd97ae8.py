"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

N = 13

# Precompute all triangle indices
_TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))


def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    # pts: (N, 2)
    i, j, k = _TRI_INDICES.T
    # Vectorized cross product for all triangles
    a = pts[i]
    b = pts[j]
    c = pts[k]
    # Area = 0.5 * |cross(b-a, c-a)|
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                   (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return 0.5 * np.min(cross)


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: shoelace on sorted points (crude)
        return 1.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _fermat_spiral(n, scale=1.0):
    """Generate n points using Fermat/Vogel spiral."""
    i = np.arange(n, dtype=float)
    r = np.sqrt(i / n) * scale
    theta = i * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _circle_init(n, radius=1.0):
    """Points on a circle with slight radial perturbation."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n) * radius
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    pts = np.column_stack((x, y))
    pts -= pts.mean(axis=0)
    return pts


def _grid_init(n):
    """Approximate grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count < n:
                pts.append([i / (side - 1), j / (side - 1)])
                count += 1
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    return pts


def _random_init(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2) * 2 - 1
    pts -= pts.mean(axis=0)
    return pts


def _local_search(pts, max_iters=5000, step_init=0.1, step_decay=0.995, 
                  step_min=1e-6, rng_seed=0, time_limit=10.0):
    """Local search with simulated annealing-style perturbations."""
    rng = np.random.RandomState(rng_seed)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = best_score
    step = step_init
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if step < step_min:
            break
        
        # Pick 1 or 2 points to move
        n_move = 1 if rng.random() < 0.6 else 2
        idx = rng.choice(N, size=n_move, replace=False)
        
        new_pts = current_pts.copy()
        for i in idx:
            delta = rng.randn(2) * step
            new_pts[i] += delta
        
        new_score = _score(new_pts)
        
        # Accept if better or with probability exp(-(new_score - current_score)/T)
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_pts = new_pts.copy()
                best_score = new_score
        else:
            # Temperature decreases over iterations
            T = 0.01 * (step / step_init)
            if T > 1e-10:
                prob = np.exp((new_score - current_score) / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
        
        step *= step_decay
    
    return best_pts, best_score


def _refine(pts, max_iters=2000, tol=1e-10):
    """High-precision refinement using gradient-free local search with tiny steps."""
    rng = np.random.RandomState(123)
    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = best_score
    step = 1e-3
    
    for it in range(max_iters):
        if step < tol:
            break
        idx = rng.randint(0, N)
        new_pts = current_pts.copy()
        delta = rng.randn(2) * step
        new_pts[idx] += delta
        new_score = _score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_pts = new_pts.copy()
                best_score = new_score
        
        step *= 0.99
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 130  # Safety margin
    
    # Generate diverse initializations
    inits = []
    inits.append(_fermat_spiral(N, scale=1.0))
    inits.append(_circle_init(N, radius=1.0))
    inits.append(_grid_init(N))
    for seed in range(5):
        inits.append(_random_init(N, seed=100 + seed))
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Local search from each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        pts, score = _local_search(
            init, 
            max_iters=3000, 
            step_init=0.15, 
            step_decay=0.997,
            step_min=1e-7,
            rng_seed=idx * 7 + 1,
            time_limit=15.0
        )
        if score > best_score:
            best_pts = pts
            best_score = score
    
    # Phase 2: Multi-start local search from perturbed best
    if best_pts is not None and time.time() < deadline - 15:
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_pts + rng.randn(N, 2) * 0.05
            pts, score = _local_search(
                perturbed,
                max_iters=2000,
                step_init=0.05,
                step_decay=0.998,
                step_min=1e-7,
                rng_seed=1000 + restart,
                time_limit=10.0
            )
            if score > best_score:
                best_pts = pts
                best_score = score
    
    # Phase 3: High-precision refinement
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_score = _refine(best_pts, max_iters=3000, tol=1e-11)
    
    if best_pts is None:
        best_pts = _fermat_spiral(N, scale=1.0)
    
    return best_pts
