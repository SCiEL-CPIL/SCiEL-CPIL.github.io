"""Heilbronn convex regions: maximize minimum triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

# EVOLVE-BLOCK-START

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle using cross product
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _fermat_spiral(n, scale=1.0):
    """Generate points on a Fermat (Vogel) spiral, centered at origin."""
    i = np.arange(n)
    r = np.sqrt(i / n) * scale
    theta = i * 2.399963229728653  # Golden angle in radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    # Center the points
    points -= np.mean(points, axis=0)
    return points

def _perturb(points, sigma):
    """Add Gaussian noise to points."""
    return points + np.random.randn(*points.shape) * sigma

def _simulated_annealing(init_points, max_iters=5000, t0=0.05, t_end=1e-6, cooling=0.9995):
    """Run simulated annealing starting from init_points."""
    current = init_points.copy()
    current_score = _score(current)
    
    best = current.copy()
    best_score = current_score
    
    # Precompute triangle index triples for efficiency
    n = N
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    triples = np.array(triples)
    
    def _min_tri_with_indices(pts):
        """Return min area and the triple that achieves it."""
        n = len(pts)
        min_area = float('inf')
        min_idx = (0, 1, 2)
        for idx in range(len(triples)):
            i, j, k = triples[idx]
            v1 = pts[j] - pts[i]
            v2 = pts[k] - pts[i]
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            if area < min_area:
                min_area = area
                min_idx = (i, j, k)
        return min_area, min_idx
    
    for it in range(max_iters):
        # Adaptive perturbation: smaller as we cool
        sigma = t0 * (t_end / t0) ** (it / max_iters)
        sigma = max(sigma, 1e-4)
        
        # Perturb one or two random points
        n_perturb = 1 if np.random.random() < 0.7 else 2
        idxs = np.random.choice(n, n_perturb, replace=False)
        
        new_pts = current.copy()
        new_pts[idxs] += np.random.randn(n_perturb, 2) * sigma
        
        new_score = _score(new_pts)
        
        # Acceptance: always accept improvements; accept worse with probability exp(-delta/T)
        if new_score > current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta = current_score - new_score
            T = t0 * (t_end / t0) ** (it / max_iters)
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current = new_pts
                current_score = new_score
    
    return best, best_score

def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_points = None
    best_score = 0.0
    
    # Multi-start with Fermat spiral variants
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Vary the scale and rotation for diversity
        scale = 1.0 + 0.1 * restart
        init = _fermat_spiral(N, scale=scale)
        
        # Add small random rotation
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        init = init @ rot.T
        
        # Run simulated annealing
        pts, score = _simulated_annealing(
            init,
            max_iters=3000,
            t0=0.03 + 0.005 * restart,
            t_end=1e-7,
            cooling=0.9998
        )
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # If we have time, do a final refinement pass on the best
    if best_points is not None and time.time() < deadline - 15:
        refined, refined_score = _simulated_annealing(
            best_points,
            max_iters=2000,
            t0=0.01,
            t_end=1e-8,
            cooling=0.9999
        )
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    if best_points is None:
        # Fallback: Fermat spiral
        best_points = _fermat_spiral(N)
    
    return best_points

# EVOLVE-BLOCK-END
