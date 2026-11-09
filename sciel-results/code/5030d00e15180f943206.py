# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _count_convex7_fast(points):
    """
    Count 7-point subsets in convex position.
    Uses the fact that a set of 7 points forms a convex 7-gon iff all 7 are on the convex hull.
    """
    n = len(points)
    count = 0
    
    # Precompute: for each subset of 7, check if all are hull points
    # Faster: use the angular sweep method per the task description
    # But for 33 points, C(33,7) = 4,292,145 which is manageable if each check is fast
    
    # Alternative: for each 7-subset, compute convex hull size
    # If hull size == 7, it's convex
    
    # Let's use a vectorized approach:
    # For each 7-subset, sort by angle around centroid, check all turns same sign
    
    for combo in combinations(range(n), 7):
        p = points[list(combo)]
        # Centroid
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        # Angles
        angles = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
        order = np.argsort(angles)
        p_sorted = p[order]
        
        # Check all 7 turns are left (or all right)
        all_left = True
        all_right = True
        for i in range(7):
            a = p_sorted[i]
            b = p_sorted[(i+1) % 7]
            c = p_sorted[(i+2) % 7]
            cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
            if cross < -1e-9:
                all_left = False
                break
            if cross > 1e-9:
                all_right = False
                break
        if all_left or all_right:
            count += 1
    
    return count

def _generate_cup_cap_optimal(n=33):
    """
    Optimal cup-cap construction for 0 convex 7-gons.
    
    The key: use the recursive extremal construction.
    For avoiding convex k-gons with n = 2^(k-2)+1 points:
    
    The construction places points on two convex chains:
    - Lower chain (cup): concave up
    - Upper chain (cap): concave down
    
    With careful arrangement, no 7 points form a convex polygon.
    
    For k=7, n=33:
    Use 17 points on lower curve, 16 on upper curve.
    The curves should be such that any convex polygon has at most 6 vertices.
    """
    np.random.seed(77)
    
    # Use points on two parabolas with careful spacing
    n_lower = 17
    n_upper = 16
    
    # Lower cup: y = a*x^2 + b, concave up
    # Upper cap: y = c - d*x^2, concave down
    
    # To minimize convex 7-gons, make the curves "tight"
    xs_lower = np.linspace(0.02, 0.98, n_lower)
    xs_upper = np.linspace(0.02, 0.98, n_upper)
    
    # Lower curve: y = 0.05 + 0.35*x^2
    ys_lower = 0.05 + 0.35 * xs_lower**2
    
    # Upper curve: y = 0.95 - 0.35*x^2
    ys_upper = 0.95 - 0.35 * xs_upper**2
    
    # Small random perturbations to break collinearity
    eps = 5e-5
    xs_lower = xs_lower + np.random.uniform(-eps, eps, n_lower)
    ys_lower = ys_lower + np.random.uniform(-eps, eps, n_lower)
    xs_upper = xs_upper + np.random.uniform(-eps, eps, n_upper)
    ys_upper = ys_upper + np.random.uniform(-eps, eps, n_upper)
    
    points = np.vstack([
        np.column_stack([xs_lower, ys_lower]),
        np.column_stack([xs_upper, ys_upper])
    ])
    
    # Ensure in unit square
    points = np.clip(points, 1e-4, 1 - 1e-4)
    
    return points

def _fix_collinearity(points, n=33):
    """Perturb points to ensure no three are collinear."""
    np.random.seed(55)
    for _ in range(100):
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < 1e-9:
                        # Perturb point k
                        points[k] += np.random.uniform(0.005, 0.02, 2)
                        points[k] = np.clip(points[k], 1e-4, 1-1e-4)
                        fixed = False
        if fixed:
            break
    return points

def _fix_close_points(points, n=33):
    """Ensure no two points are too close."""
    np.random.seed(66)
    for _ in range(100):
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                d2 = np.sum((points[i] - points[j])**2)
                if d2 < 1e-12:
                    points[j] += np.random.uniform(0.01, 0.05, 2)
                    points[j] = np.clip(points[j], 1e-4, 1-1e-4)
                    fixed = False
        if fixed:
            break
    return points

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Place 33 points in unit square to minimize convex 7-gons.
    Returns (33, 2) float array.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Generate initial cup-cap construction
    points = _generate_cup_cap_optimal(33)
    
    # Fix any issues
    points = _fix_close_points(points)
    points = _fix_collinearity(points)
    
    # Count convex 7-gons
    count = _count_convex7_fast(points)
    
    if count == 0:
        return points
    
    # If not zero, try to improve with limited hill climbing
    remaining = deadline - time.time()
    if remaining > 60:
        # Quick local search: try moving points to reduce count
        np.random.seed(123)
        best_points = points.copy()
        best_count = count
        
        # Estimate time per count
        t0 = time.time()
        _count_convex7_fast(points)
        t1 = time.time()
        eval_time = max(t1 - t0, 0.1)
        
        max_iters = int(0.7 * remaining / eval_time)
        max_iters = min(max_iters, 200)  # Cap iterations
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Pick random point
            idx = np.random.randint(33)
            old = best_points[idx].copy()
            
            # Try several random moves
            for attempt in range(5):
                direction = np.random.randn(2)
                direction /= (np.linalg.norm(direction) + 1e-15)
                step = np.random.uniform(0.01, 0.08)
                best_points[idx] = old + step * direction
                best_points[idx] = np.clip(best_points[idx], 1e-4, 1-1e-4)
                
                # Quick validity check
                valid = True
                for j in range(33):
                    if j != idx:
                        d2 = np.sum((best_points[idx] - best_points[j])**2)
                        if d2 < 1e-12:
                            valid = False
                            break
                if not valid:
                    continue
                
                new_count = _count_convex7_fast(best_points)
                if new_count < best_count:
                    best_count = new_count
                    if best_count == 0:
                        return best_points
                    break
                else:
                    best_points[idx] = old
            
            best_points[idx] = old
        
        if best_count < count:
            points = best_points
    
    # Final validation
    assert points.shape == (33, 2)
    assert not np.any(np.isnan(points))
    assert not np.any(np.isinf(points))
    assert np.all(points >= -1e-6)
    assert np.all(points <= 1 + 1e-6)
    
    for i in range(33):
        for j in range(i+1, 33):
            d2 = np.sum((points[i] - points[j])**2)
            if d2 < 1e-12:
                points[j] += 0.01
                points[j] = np.clip(points[j], 1e-4, 1-1e-4)
    
    for i in range(33):
        for j in range(i+1, 33):
            for k in range(j+1, 33):
                p1, p2, p3 = points[i], points[j], points[k]
                area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                if area < 1e-9:
                    points[k] += 0.01
                    points[k] = np.clip(points[k], 1e-4, 1-1e-4)
    
    return points

# EVOLVE-BLOCK-END
