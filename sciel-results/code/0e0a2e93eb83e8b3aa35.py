# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min):
        return 0.0
    return a_min / a_hull

def _regular_13gon():
    """Generate a regular 13-gon centered at origin with radius 1."""
    n = 13
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _fourier_perturb(base_points, amplitudes, phases, modes):
    """
    Perturb a regular polygon using low-order Fourier modes.
    Each point's radius is modulated by a sum of cosine terms.
    """
    n = len(base_points)
    angles = np.arctan2(base_points[:, 1], base_points[:, 0])
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    radii = np.ones(n)
    for m, amp, phase in zip(modes, amplitudes, phases):
        radii += amp * np.cos(m * angles + phase)
    
    # Ensure radii stay positive
    radii = np.maximum(radii, 0.3)
    
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _objective(params, base_points, modes):
    """Objective function to minimize (negative of score)."""
    n_params = len(params)
    n_modes = len(modes)
    amplitudes = params[:n_modes]
    phases = params[n_modes:2 * n_modes]
    
    points = _fourier_perturb(base_points, amplitudes, phases, modes)
    
    # Check for validity
    if not np.all(np.isfinite(points)):
        return 1e6
    
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull <= 0 or not np.isfinite(a_min) or a_min <= 0:
        return 1e6
    
    return -(a_min / a_hull)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580 seconds, leaving margin before 630s limit
    
    # Start with regular 13-gon
    base_points = _regular_13gon()
    best_points = base_points.copy()
    best_score = _score(best_points)
    
    # Use Fourier modes: 2, 3, 4, 5 (low-order harmonics)
    modes = [2, 3, 4, 5]
    n_modes = len(modes)
    
    # Initial parameters: zero amplitudes, zero phases
    x0 = np.zeros(2 * n_modes)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            _objective,
            x0,
            args=(base_points, modes),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline and result.fun < -best_score:
            params = result.x
            amplitudes = params[:n_modes]
            phases = params[n_modes:2 * n_modes]
            candidate = _fourier_perturb(base_points, amplitudes, phases, modes)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_points = candidate
                best_score = cand_score
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a few random restarts with small perturbations
        rng = np.random.default_rng(seed=123)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Small random amplitudes and phases
            amps = rng.uniform(-0.1, 0.1, n_modes)
            phs = rng.uniform(0, 2 * np.pi, n_modes)
            candidate = _fourier_perturb(base_points, amps, phs, modes)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_points = candidate
                best_score = cand_score
    
    return best_points

# EVOLVE-BLOCK-END
