import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations

N = 13

# Pre-compute triangle indices for N=13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)


def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area


def _bottleneck_triangle(points):
    """Return indices (i,j,k) and area of the minimum-area triangle."""
    areas = _triangle_areas(points)
    min_idx = np.argmin(areas)
    return _i_arr[min_idx], _j_arr[min_idx], _k_arr[min_idx], areas[min_idx]


def _hull_indices(points):
    """Return set of indices on the convex hull."""
    try:
        hull = spat.ConvexHull(points)
        return set(hull.vertices)
    except Exception:
        return set(range(N))


def _make_move(points, rng):
    """
    Generate a candidate move. Returns new_points or None if invalid.
    Uses a mix of strategies:
      1. Small Gaussian perturbation of a random point
      2. Rotate+scale hull points around centroid
      3. Push bottleneck triangle vertices apart
      4. Swap two random points
    """
    new_pts = points.copy()
    strategy = rng.random()
    
    if strategy < 0.35:
        # Gaussian perturbation of 1-3 random points
        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        scale = 0.01 + 0.05 * rng.random()
        for idx in idxs:
            new_pts[idx] += rng.normal(0, scale, 2)
            
    elif strategy < 0.55:
        # Rotate + scale hull points around centroid
        centroid = np.mean(points, axis=0)
        hull_pts = _hull_indices(points)
        if len(hull_pts) < 3:
            return None
        angle = rng.normal(0, 0.05)
        scale = 1.0 + rng.normal(0, 0.03)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        for idx in hull_pts:
            p = points[idx] - centroid
            new_pts[idx] = centroid + scale * (R @ p)
            
    elif strategy < 0.80:
        # Push bottleneck triangle vertices apart
        i, j, k, _ = _bottleneck_triangle(points)
        p_i, p_j, p_k = points[i], points[j], points[k]
        centroid_jk = (p_j + p_k) / 2.0
        centroid_ik = (p_i + p_k) / 2.0
        centroid_ij = (p_i + p_j) / 2.0
        
        dir_i = p_i - centroid_jk
        dir_j = p_j - centroid_ik
        dir_k = p_k - centroid_ij
        
        norm_i = np.linalg.norm(dir_i)
        norm_j = np.linalg.norm(dir_j)
        norm_k = np.linalg.norm(dir_k)
        
        if norm_i > 1e-8:
            new_pts[i] = p_i + 0.02 * dir_i / norm_i
        if norm_j > 1e-8:
            new_pts[j] = p_j + 0.02 * dir_j / norm_j
        if norm_k > 1e-8:
            new_pts[k] = p_k + 0.02 * dir_k / norm_k
            
    else:
        # Swap two random points
        idxs = rng.choice(N, size=2, replace=False)
        a, b = idxs
        new_pts[a], new_pts[b] = points[b].copy(), points[a].copy()
    
    return new_pts


def _simulated_annealing(points, time_budget, seed=0):
    """
    Run simulated annealing on the exact (non-smooth) objective.
    Returns (best_points, best_score).
    """
    rng = np.random.default_rng(seed)
    
    current_pts = points.copy()
    current_score = calculate_score(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Cooling schedule: start high, cool geometrically
    T_start = 0.05
    T_end = 1e-6
    n_iter = 50000
    
    # Adaptive: estimate iterations per second
    t0 = time.time()
    
    for it in range(n_iter):
        if time.time() - t0 > time_budget:
            break
        
        # Temperature: geometric cooling
        progress = it / n_iter
        T = T_start * (T_end / T_start) ** progress
        
        # Generate move
        trial_pts = _make_move(current_pts, rng)
        if trial_pts is None:
            continue
        
        trial_score = calculate_score(trial_pts)
        
        # Metropolis acceptance
        delta = trial_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_pts = trial_pts
            current_score = trial_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
    
    return best_pts, best_score


def _local_polish(points, max_iter=2000):
    """
    Quick local polish: greedily push bottleneck triangle apart.
    Returns (best_points, best_score).
    """
    pts = points.copy()
    score = calculate_score(pts)
    best_pts = pts.copy()
    best_score = score
    
    step = 0.01
    for it in range(max_iter):
        i, j, k, min_area = _bottleneck_triangle(pts)
        p_i, p_j, p_k = pts[i], pts[j], pts[k]
        
        centroid_jk = (p_j + p_k) / 2.0
        centroid_ik = (p_i + p_k) / 2.0
        centroid_ij = (p_i + p_j) / 2.0
        
        dir_i = p_i - centroid_jk
        dir_j = p_j - centroid_ik
        dir_k = p_k - centroid_ij
        
        norm_i = np.linalg.norm(dir_i)
        norm_j = np.linalg.norm(dir_j)
        norm_k = np.linalg.norm(dir_k)
        
        if norm_i > 1e-8:
            trial = pts.copy()
            trial[i] = p_i + step * dir_i / norm_i
            trial[j] = p_j + step * dir_j / norm_j
            trial[k] = p_k + step * dir_k / norm_k
            trial_score = calculate_score(trial)
            
            if trial_score > score:
                pts = trial
                score = trial_score
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
            else:
                step *= 0.9
                if step < 1e-10:
                    break
        else:
            break
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial configurations
    np.random.seed(42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Perturbed regular
    noise = np.random.normal(0, 0.15, (N, 2))
    perturbed = regular + noise
    
    # 3. Fibonacci spiral
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N)
        theta = golden_angle * i
        fib[i] = [r * np.cos(theta), r * np.sin(theta)]
    fib /= np.max(np.linalg.norm(fib, axis=1))
    
    # 4. Two-ring: inner 6 + outer 7
    inner_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    two_ring = np.vstack([
        np.column_stack((0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles))),
        np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    ])
    
    # 5. Three-ring: center + 4 inner + 8 outer
    center = np.array([[0.0, 0.0]])
    inner4_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer8_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    three_ring = np.vstack([
        center,
        np.column_stack((0.4 * np.cos(inner4_angles), 0.4 * np.sin(inner4_angles))),
        np.column_stack((np.cos(outer8_angles), np.sin(outer8_angles)))
    ])
    
    # 6. Random points in unit disk
    random_pts = np.random.uniform(-1, 1, (N, 2))
    norms = np.linalg.norm(random_pts, axis=1)
    random_pts[norms > 1] = random_pts[norms > 1] / norms[norms > 1][:, None]
    
    # 7. Another perturbed regular
    np.random.seed(123)
    noise2 = np.random.normal(0, 0.08, (N, 2))
    perturbed2 = regular + noise2
    
    # 8. Spiral
    t = np.linspace(0, 3 * np.pi, N)
    r = t / (3 * np.pi)
    spiral = np.column_stack((r * np.cos(t), r * np.sin(t)))
    
    candidates = [regular, perturbed, fib, two_ring, three_ring, random_pts, perturbed2, spiral]
    
    # First pass: score all candidates
    scored = []
    for init in candidates:
        s = calculate_score(init)
        scored.append((s, init))
    
    # Sort by score descending
    scored.sort(key=lambda x: -x[0])
    
    # Run SA on top candidates, with time budget for each
    n_sa_runs = 4
    sa_budget_each = (deadline - start_time - 10.0) / n_sa_runs
    sa_budget_each = max(sa_budget_each, 15.0)
    
    for rank in range(min(n_sa_runs, len(scored))):
        if time.time() > deadline - 10.0:
            break
        
        s, init = scored[rank]
        if s > best_score:
            best_score = s
            best_points = init.copy()
        
        # Run SA
        seed = 42 + rank * 1000
        sa_pts, sa_score = _simulated_annealing(init, sa_budget_each, seed=seed)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
        
        # Quick local polish on SA result
        if time.time() < deadline - 5.0:
            polished, polish_score = _local_polish(sa_pts, max_iter=500)
            if polish_score > best_score:
                best_score = polish_score
                best_points = polished.copy()
    
    # Final polish on the best points
    if best_points is not None and time.time() < deadline - 2.0:
        final_pts, final_score = _local_polish(best_points, max_iter=1000)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    if best_points is None:
        return np.random.rand(N, 2)
    
    return best_points
