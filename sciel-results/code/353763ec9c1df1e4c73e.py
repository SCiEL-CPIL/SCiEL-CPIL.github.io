"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 145.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def construct():
    start_time = time.time()
    rng = np.random.RandomState(12345)
    
    # Start from a good initial configuration
    current_pts = _initial_regular_polygon(N)
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Simulated annealing parameters
    T_start = 0.05  # Initial temperature (step size scale)
    T_end = 0.001   # Final temperature
    alpha = 0.9995  # Cooling rate per step
    max_iter = 50000
    kick_interval = 5000  # Every N steps, do a bigger perturbation
    
    T = T_start
    step = 0
    
    for step in range(max_iter):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate perturbation
        if step % kick_interval == 0 and step > 0:
            # Kick: larger perturbation to escape local minima
            noise_scale = T * 3.0
        else:
            noise_scale = T
        
        # Perturb a random subset of points (sparse updates for speed)
        n_perturb = rng.randint(1, 4)  # Perturb 1-3 points
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        new_pts = current_pts.copy()
        for idx in indices:
            new_pts[idx] += rng.normal(0, noise_scale, size=2)
        
        # Evaluate new configuration
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            # Better solution: always accept
            current_pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        else:
            # Worse solution: accept with probability exp((new - current) / T)
            delta = current_score - new_score
            if delta > 0:
                prob = np.exp(-delta / (T + 1e-12))
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
        
        # Cool down
        T = max(T_end, T * alpha)
    
    # Final local polish: fine-tune with very small steps
    if time.time() - start_time < DEADLINE - 3:
        polish_T = 0.0005
        for _ in range(5000):
            if time.time() - start_time > DEADLINE - 2:
                break
            n_perturb = 1
            idx = rng.randint(0, N)
            new_pts = best_pts.copy()
            new_pts[idx] += rng.normal(0, polish_T, size=2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
