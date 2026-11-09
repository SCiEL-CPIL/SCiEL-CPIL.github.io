# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

def _score_fast(points):
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    n = len(points)
    # Precompute all triangle indices
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Only keep unique triples i < j < k
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_flat = i_flat[mask]
    j_flat = j_flat[mask]
    k_flat = k_flat[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _score(points):
    """Compute A_min / A_hull for a set of points."""
    return _score_fast(points)

def _negative_score_flat(flat_points):
    """Objective function for differential evolution (minimize negative score)."""
    points = flat_points.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score_fast(points)
    if s <= 0:
        return 1e10
    return -s

def _local_search(points, rng, deadline, max_iter=5000):
    """Adaptive local search with simulated annealing-like acceptance."""
    current = points.copy()
    current_score = _score_fast(current)
    if current_score <= 0:
        return current, 0.0
    
    best = current.copy()
    best_score = current_score
    
    # Adaptive step size
    step_size = 0.05
    temperature = 0.01
    cooldown = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a random point to move
        i = rng.integers(0, 13)
        
        # Generate random direction
        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction /= norm
        
        # Try multiple step sizes
        trial = current.copy()
        trial[i] = current[i] + step_size * direction
        trial_score = _score_fast(trial)
        
        if trial_score > current_score:
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
            step_size *= 1.1  # Increase step if improving
        else:
            # Simulated annealing acceptance
            delta = trial_score - current_score
            if delta < 0 and rng.random() < np.exp(delta / temperature):
                current = trial
                current_score = trial_score
            step_size *= 0.95  # Decrease step if not improving
        
        temperature *= cooldown
        step_size = max(step_size, 0.001)
    
    return best, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using multi-stage optimization.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 13
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(seed=12345)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Hexagonal lattice pattern
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 2. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_pts)
    
    # 3. Grid-based
    grid_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Fibonacci spiral
    fib_pts = np.zeros((n, 2))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        r = np.sqrt(i + 1) * 0.3
        theta = i * golden_angle
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    initializations.append(fib_pts)
    
    # 5. Random points in unit square
    for _ in range(8):
        initializations.append(rng.random((n, 2)) * 2 - 1)
    
    # 6. Concentric rings
    ring1 = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 5, endpoint=False)),
                             np.sin(np.linspace(0, 2*np.pi, 5, endpoint=False))])
    ring2 = np.column_stack([np.cos(np.linspace(np.pi/5, 2*np.pi, 5, endpoint=False))*1.5,
                             np.sin(np.linspace(np.pi/5, 2*np.pi, 5, endpoint=False))*1.5])
    center = np.array([[0.0, 0.0]])
    ring_pts = np.vstack([center, ring1, ring2])
    initializations.append(ring_pts)
    
    # Evaluate all initializations
    for init_pts in initializations:
        if time.time() > deadline - 120:
            break
        s = _score_fast(init_pts)
        update_best(init_pts, s)
    
    # Run differential evolution from best initialization
    if best_points is not None and time.time() < deadline - 90:
        bounds = [(-2, 2)] * (2 * n)
        best_init = best_points.flatten()
        
        # Try multiple DE configurations
        for de_seed in [42, 123, 456]:
            if time.time() > deadline - 60:
                break
            
            try:
                result = differential_evolution(
                    _negative_score_flat,
                    bounds,
                    maxiter=80,
                    popsize=25,
                    tol=1e-5,
                    mutation=(0.5, 1.5),
                    recombination=0.7,
                    seed=de_seed,
                    polish=False,
                    init='sobol',
                    x0=best_init,
                    updating='immediate',
                    callback=None
                )
                
                if result.fun < 0:
                    de_points = result.x.reshape(n, 2)
                    s = _score_fast(de_points)
                    if s > 0:
                        update_best(de_points, s)
            except Exception:
                pass
    
    # Local search from best point
    if best_points is not None and time.time() < deadline - 30:
        local_result, local_score = _local_search(best_points, rng, deadline - 10, max_iter=3000)
        update_best(local_result, local_score)
    
    # Additional local search rounds from diverse starting points
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        
        # Start from a random perturbation of best
        perturbed = best_points + rng.normal(size=(n, 2)) * 0.1
        local_result, local_score = _local_search(perturbed, rng, deadline - 10, max_iter=2000)
        update_best(local_result, local_score)
    
    # Final validation
    if best_points is None or not np.all(np.isfinite(best_points)):
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    final_score = _score_fast(best_points)
    if final_score <= 0:
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
