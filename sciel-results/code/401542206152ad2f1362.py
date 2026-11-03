"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def _all_tri_indices(n):
    """Precompute all triple indices."""
    return list(itertools.combinations(range(n), 3))

def _triangle_area(points, i, j, k):
    """Area of triangle formed by points i, j, k."""
    a, b, c = points[i], points[j], points[k]
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

def _convex_hull_area(points):
    """Compute convex hull area using monotone chain (Andrew's algorithm)."""
    n = len(points)
    if n < 3:
        return 0.0
    points = np.array(points, dtype=float)
    points = np.sort(points, axis=0)
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return area

def _compute_min_tri_area_norm(points):
    """Compute minimum triangle area normalized by convex hull area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points, i, j, k)
                if a < min_area:
                    min_area = a
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    return min_area / hull_area

def _soft_min_triangle_ratio(points, triples, hull_area, beta=100.0):
    """
    Smooth surrogate for min triangle area / hull area.
    Uses log-sum-exp for soft minimum of triangle areas.
    """
    n = len(points)
    areas = np.empty(len(triples))
    for idx, (i, j, k) in enumerate(triples):
        a, b, c = points[i], points[j], points[k]
        areas[idx] = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    
    # Soft minimum using log-sum-exp trick
    # min(areas) ≈ -log(sum(exp(-beta * areas))) / beta
    # For numerical stability:
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_area = -log_sum / beta
    
    if hull_area < 1e-12:
        return 0.0
    
    return soft_min_area / hull_area

def _objective_flat(x, triples, beta=100.0):
    """
    Objective function for L-BFGS-B: minimize -soft_min_triangle_ratio.
    x is flat array of shape (2*n,).
    """
    n = len(triples) // 1  # Just for clarity, n points
    n_points = len(x) // 2
    points = x.reshape(n_points, 2)
    
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1.0  # Penalize degenerate hulls
    
    ratio = _soft_min_triangle_ratio(points, triples, hull_area, beta)
    return -ratio

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _halton_sequence(n, dims=2):
    """Generate Halton sequence points in [0,1]^dims."""
    points = np.zeros((n, dims))
    for dim in range(dims):
        base = dim + 2  # 2, 3, 5, ...
        f = 1.0 / base
        for i in range(n):
            f /= base
            points[i, dim] = f
    return points

def _generate_diverse_starts(n, num_starts=8):
    """Generate diverse starting configurations."""
    starts = []
    
    # 1. Regular polygon
    starts.append(_regular_polygon(n, 1.0))
    
    # 2. Regular polygon with slight perturbation
    base = _regular_polygon(n, 1.0)
    np.random.seed(123)
    perturbed = base + np.random.normal(0, 0.05, size=base.shape)
    starts.append(perturbed)
    
    # 3. Halton-based points scaled to unit area
    halton = _halton_sequence(n)
    # Center and scale
    halton_centered = halton - halton.mean(axis=0)
    hull_a = _convex_hull_area(halton_centered)
    if hull_a > 1e-12:
        halton_scaled = halton_centered / np.sqrt(hull_a)
        starts.append(halton_scaled)
    else:
        starts.append(_regular_polygon(n, 1.0))
    
    # 4. Another Halton with different seed
    np.random.seed(456)
    halton2 = _halton_sequence(n) + np.random.normal(0, 0.1, size=(n, 2))
    halton2_centered = halton2 - halton2.mean(axis=0)
    hull_a2 = _convex_hull_area(halton2_centered)
    if hull_a2 > 1e-12:
        halton2_scaled = halton2_centered / np.sqrt(hull_a2)
        starts.append(halton2_scaled)
    else:
        starts.append(_regular_polygon(n, 1.0))
    
    # 5. Random points on circle with perturbation
    np.random.seed(789)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, n))
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(circle_pts)
    
    # 6. More random angles
    np.random.seed(101)
    angles2 = np.sort(np.random.uniform(0, 2 * np.pi, n))
    circle_pts2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    starts.append(circle_pts2)
    
    # 7. Regular polygon with radius variation
    for radius in [0.8, 1.2]:
        starts.append(_regular_polygon(n, radius))
    
    return starts[:num_starts]

def _optimize_from_start(points, triples, beta=100.0, maxiter=300):
    """Optimize from a given starting configuration."""
    x0 = points.flatten()
    
    try:
        res = minimize(
            _objective_flat,
            x0,
            args=(triples, beta),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        opt_points = res.x.reshape(-1, 2)
        score = _compute_min_tri_area_norm(opt_points)
        return opt_points, score, res
    except Exception as e:
        return points, _compute_min_tri_area_norm(points), None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n = N
    triples = _all_tri_indices(n)
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse starting points
    starts = _generate_diverse_starts(n, num_starts=10)
    
    # Try each starting point
    for s_idx, start_pts in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        
        # Evaluate starting score
        start_score = _compute_min_tri_area_norm(start_pts)
        if start_score > best_score:
            best_score = start_score
            best_points = start_pts.copy()
        
        # Optimize from this start
        opt_pts, opt_score, res = _optimize_from_start(start_pts, triples, beta=100.0, maxiter=300)
        
        if time.time() > deadline - 5.0:
            break
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
        
        # Try with different beta values for additional diversity
        if time.time() > deadline - 15.0:
            break
        
        for beta in [50.0, 200.0]:
            if time.time() > deadline - 5.0:
                break
            opt_pts2, opt_score2, _ = _optimize_from_start(start_pts, triples, beta=beta, maxiter=200)
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_pts2.copy()
    
    # If no valid points found, use regular polygon
    if best_points is None:
        best_points = _regular_polygon(n, 1.0)
    
    return best_points
# EVOLVE-BLOCK-END
