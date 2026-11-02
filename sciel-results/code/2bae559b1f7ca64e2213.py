"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import random
import time
import itertools

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Triangle area = 0.5 * |cross product|
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # For 2D, volume is area
    except Exception:
        # Fallback: if all points are collinear or degenerate
        return 1e-10

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _fermat_spiral(n, scale=1.0):
    """Generate points using Fermat spiral (Vogel's model)."""
    i = np.arange(n)
    r = np.sqrt(i / n) * scale
    theta = i * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    pts -= np.mean(pts, axis=0)
    return pts

def _circle_init(n, scale=1.0):
    """Generate points on a circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = scale * np.cos(angles)
    y = scale * np.sin(angles)
    return np.column_stack((x, y))

def _grid_init(n, scale=1.0):
    """Generate points on a grid-like pattern."""
    # Arrange in a roughly square grid
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    pts = []
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            x = (j - (cols-1)/2) * scale / max(cols-1, 1)
            y = (i - (rows-1)/2) * scale / max(rows-1, 1)
            pts.append([x, y])
            count += 1
    pts = np.array(pts[:n])
    pts -= np.mean(pts, axis=0)
    return pts

def _random_init(n, scale=1.0, seed=42):
    """Generate random points in a square."""
    rng = random.Random(seed)
    pts = np.array([[rng.uniform(-scale, scale), rng.uniform(-scale, scale)] for _ in range(n)])
    pts -= np.mean(pts, axis=0)
    return pts

def _sa_optimize(init_pts, time_limit, rng_seed=0):
    """Simulated annealing optimization."""
    rng = random.Random(rng_seed)
    n = len(init_pts)
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    
    def fast_score(pts):
        """Fast score computation using precomputed indices."""
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            return 0.0
        # Vectorized triangle area computation
        idx = tri_indices
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        areas = 0.5 * np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0]))
        min_tri = np.min(areas)
        return min_tri / hull_area
    
    current = init_pts.copy()
    current_score = fast_score(current)
    best = current.copy()
    best_score = current_score
    
    # SA parameters
    t0 = time.time()
    deadline = t0 + time_limit
    T = 0.01
    T_min = 1e-8
    alpha = 0.9995
    
    step_size = 0.1
    
    while time.time() < deadline - 2.0:  # Safety margin
        # Generate a move
        move_type = rng.random()
        new_pts = current.copy()
        
        if move_type < 0.5:
            # 1-point move: move one point
            idx = rng.randrange(n)
            new_pts[idx, 0] += rng.gauss(0, step_size)
            new_pts[idx, 1] += rng.gauss(0, step_size)
        elif move_type < 0.8:
            # 2-point move: move two points
            idx1, idx2 = rng.sample(range(n), 2)
            new_pts[idx1, 0] += rng.gauss(0, step_size)
            new_pts[idx1, 1] += rng.gauss(0, step_size)
            new_pts[idx2, 0] += rng.gauss(0, step_size)
            new_pts[idx2, 1] += rng.gauss(0, step_size)
        else:
            # Swap-like move: exchange positions partially
            idx1, idx2 = rng.sample(range(n), 2)
            swap_frac = rng.uniform(0.1, 0.5)
            old1 = new_pts[idx1].copy()
            old2 = new_pts[idx2].copy()
            new_pts[idx1] = old1 + swap_frac * (old2 - old1)
            new_pts[idx2] = old2 + swap_frac * (old1 - old2)
        
        new_score = fast_score(new_pts)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Adapt step size
        if rng.random() < 0.01:
            step_size *= rng.choice([0.8, 1.2])
            step_size = np.clip(step_size, 0.01, 0.5)
    
    return best, best_score

def construct():
    t0 = time.time()
    total_deadline = t0 + 140.0  # Safety margin before 150s
    
    # Multi-start with diverse initializations
    initializations = [
        ("fermat", _fermat_spiral(N, 1.0)),
        ("circle", _circle_init(N, 1.0)),
        ("grid", _grid_init(N, 1.0)),
        ("random1", _random_init(N, 1.0, seed=42)),
        ("random2", _random_init(N, 1.0, seed=123)),
        ("fermat_scaled", _fermat_spiral(N, 1.2)),
    ]
    
    best_overall = None
    best_score_overall = 0.0
    
    # Allocate time to each initialization
    time_per_init = 20.0  # 20s per init, 6 inits = 120s, plus overhead
    
    for name, init_pts in initializations:
        if time.time() > total_deadline - 10.0:
            break
        
        remaining = total_deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        
        sa_time = min(time_per_init, remaining)
        result_pts, result_score = _sa_optimize(init_pts, sa_time, rng_seed=hash(name) % 10000)
        
        if result_score > best_score_overall:
            best_score_overall = result_score
            best_overall = result_pts
    
    # If we have time, do a final refinement on the best
    if best_overall is not None and time.time() < total_deadline - 5.0:
        remaining = total_deadline - time.time() - 3.0
        if remaining > 3.0:
            refined_pts, refined_score = _sa_optimize(best_overall, remaining, rng_seed=99999)
            if refined_score > best_score_overall:
                best_overall = refined_pts
                best_score_overall = refined_score
    
    if best_overall is None:
        # Fallback: return Fermat spiral
        best_overall = _fermat_spiral(N, 1.0)
    
    return best_overall

# EVOLVE-BLOCK-END
