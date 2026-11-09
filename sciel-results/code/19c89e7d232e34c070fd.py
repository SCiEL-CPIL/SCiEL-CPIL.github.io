# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

# Deadline guard: keep a safety margin before the 630s wall-clock limit
_DEADLINE = 630.0 - 30.0  # 600s effective budget
_START_TIME = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _triangle_areas_vectorized(points):
    """Compute all triangle areas vectorized. Returns (n_tri, 3) indices and areas."""
    n = len(points)
    idx_i, idx_j, idx_k = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
    
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    
    return idx_i, idx_j, idx_k, areas


def _score(points):
    """Compute A_min / A_hull for the given points."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-15:
        return 0.0
    
    _, _, _, areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _simulated_annealing(points, max_iter=5000, T_init=0.05, T_final=0.001, cooling=0.9995):
    """Optimize using simulated annealing to escape local minima."""
    points = points.copy().astype(float)
    n = len(points)
    
    best_score = _score(points)
    best_points = points.copy()
    
    rng = np.random.default_rng(seed=42)
    T = T_init
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        
        # Generate a perturbation
        perturbation = rng.normal(0, T * 0.5, size=(n, 2))
        new_points = points + perturbation
        
        # Keep points within reasonable bounds
        new_points = np.clip(new_points, -0.1, 1.1)
        
        if not np.all(np.isfinite(new_points)):
            continue
        
        new_score = _score(new_points)
        current_score = _score(points)
        
        # Accept if better or with probability exp((new_score - current_score) / T)
        if new_score > current_score or (T > 0 and rng.random() < np.exp((new_score - current_score) / T)):
            points = new_points.copy()
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        
        # Cool down
        T = max(T_final, T * cooling)
    
    return best_points


def _local_search_refine(points, n_points, max_iter=500):
    """Refine using local search with random perturbations."""
    points = points.copy().astype(float)
    n = n_points
    
    best_score = _score(points)
    best_points = points.copy()
    
    rng = np.random.default_rng(seed=42)
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        
        improved = False
        
        for pt_idx in range(n):
            if _time_left() < 30:
                break
            
            original = points[pt_idx].copy()
            
            # Try several random perturbations with decreasing step sizes
            for step_size in [0.02, 0.01, 0.005, 0.002]:
                for attempt in range(5):
                    if _time_left() < 30:
                        break
                    
                    perturbation = rng.normal(0, step_size, size=2)
                    test_points = points.copy()
                    test_points[pt_idx] = original + perturbation
                    
                    if not np.all(np.isfinite(test_points)):
                        continue
                    if np.any(test_points < -0.5) or np.any(test_points > 1.5):
                        continue
                    
                    test_score = _score(test_points)
                    if test_score > best_score:
                        best_score = test_score
                        best_points = test_points.copy()
                        points = test_points.copy()
                        improved = True
                        break
                
                if improved:
                    break
        
        if not improved:
            break
    
    return best_points


def _init_triangular_lattice():
    """Initialize with a triangular lattice subset."""
    n = 13
    # Triangular lattice with spacing s
    s = 0.25
    points = []
    
    # Generate lattice points and select 13 that form a good configuration
    # Use a 4x4 grid of triangular lattice points
    for i in range(4):
        for j in range(4):
            x = 0.3 + i * s + (j % 2) * s / 2
            y = 0.3 + j * s * np.sqrt(3) / 2
            points.append([x, y])
    
    # Select 13 points from the 16 available
    # Use a heuristic: pick points that maximize minimum distance
    points = np.array(points[:13])
    
    # Normalize to fit in [0, 1]
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    
    if max_x - min_x > 1e-10:
        points[:, 0] = 0.1 + 0.8 * (points[:, 0] - min_x) / (max_x - min_x)
    if max_y - min_y > 1e-10:
        points[:, 1] = 0.1 + 0.8 * (points[:, 1] - min_y) / (max_y - min_y)
    
    return points


def _init_hexagonal():
    """Initialize with a hexagonal packing pattern."""
    n = 13
    # Center point + 6 in first ring + 6 in second ring (partial)
    center = [0.5, 0.5]
    r1 = 0.2
    r2 = 0.35
    
    points = [center]
    
    # First ring: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        points.append([center[0] + r1 * np.cos(angle), center[1] + r1 * np.sin(angle)])
    
    # Second ring: 6 more points (total 13)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([center[0] + r2 * np.cos(angle), center[1] + r2 * np.sin(angle)])
    
    return np.array(points)


def _init_pentagonal():
    """Initialize with a pentagonal symmetry pattern."""
    n = 13
    # Center + 5 in first ring + 5 in second ring + 2 extra
    center = [0.5, 0.5]
    r1 = 0.2
    r2 = 0.35
    
    points = [center]
    
    # First ring: 5 points (pentagon)
    for i in range(5):
        angle = i * 2 * np.pi / 5
        points.append([center[0] + r1 * np.cos(angle), center[1] + r1 * np.sin(angle)])
    
    # Second ring: 5 points (rotated pentagon)
    for i in range(5):
        angle = i * 2 * np.pi / 5 + np.pi / 5
        points.append([center[0] + r2 * np.cos(angle), center[1] + r2 * np.sin(angle)])
    
    # Add 2 more points to make 13
    points.append([0.5, 0.15])
    points.append([0.5, 0.85])
    
    return np.array(points)


def _init_fibonacci():
    """Initialize using Fibonacci distribution."""
    n = 13
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = []
    
    for i in range(n):
        angle = i * golden_angle
        r = 0.5 * np.sqrt((i + 0.5) / n)
        points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    
    return np.array(points)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    best_points = None
    best_score = -1
    
    # Generate several initial candidates
    candidates = []
    
    # Candidate 1: Triangular lattice
    candidates.append(_init_triangular_lattice())
    
    # Candidate 2: Hexagonal packing
    candidates.append(_init_hexagonal())
    
    # Candidate 3: Pentagonal symmetry
    candidates.append(_init_pentagonal())
    
    # Candidate 4: Fibonacci distribution
    candidates.append(_init_fibonacci())
    
    # Candidate 5: Random
    rng = np.random.default_rng(seed=42)
    candidates.append(0.1 + 0.8 * rng.random((n, 2)))
    
    # Candidate 6: Another random
    rng2 = np.random.default_rng(seed=123)
    candidates.append(0.1 + 0.8 * rng2.random((n, 2)))
    
    # Optimize each candidate using simulated annealing
    for candidate in candidates:
        if _time_left() < 30:
            break
        
        # Simulated annealing optimization
        optimized = _simulated_annealing(candidate, max_iter=8000)
        score = _score(optimized)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
        
        # Local search refinement if we have time
        if _time_left() > 60:
            refined = _local_search_refine(optimized, n, max_iter=300)
            score2 = _score(refined)
            if score2 > best_score:
                best_score = score2
                best_points = refined.copy()
    
    # Final fallback: if all failed, return the best we have
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = np.random.default_rng(0).random((n, 2))
    
    return best_points


# EVOLVE-BLOCK-END
