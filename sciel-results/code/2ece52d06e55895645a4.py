"""Rewritten program for Heilbronn convex regions n=14."""
import numpy as np
from scipy.optimize import minimize, differential_evolution, LinearConstraint
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """Compute areas of all triangles formed by triples of points.
    Returns an array of shape (C(N,3),) with actual areas."""
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas.append(abs(cross) * 0.5)
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1e-10


def _score(points):
    """Compute the score: minimum triangle area / convex hull area."""
    tri_areas = _triangle_areas(points)
    min_tri = np.min(tri_areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_tri / hull_area


def _soft_min(values, temperature=0.01):
    """Soft minimum approximation using log-sum-exp trick."""
    scaled = -values / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum


def _objective_softmin(points, temperature=0.05):
    """Objective to minimize: negative soft-min of triangle areas, normalized by hull area."""
    tri_areas = _triangle_areas(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    normalized = tri_areas / hull_area
    soft_min_val = _soft_min(normalized, temperature)
    return -soft_min_val


def _polar_to_cartesian(radii, angles):
    """Convert polar coordinates (radii, angles) to Cartesian coordinates.
    Ensures strictly increasing angular order for convex position."""
    # Sort by angle to ensure convex ordering
    order = np.argsort(angles)
    sorted_radii = radii[order]
    sorted_angles = angles[order]
    
    # Ensure angles are in [0, 2*pi)
    sorted_angles = sorted_angles % (2 * np.pi)
    
    # If angles are too close, spread them out
    min_angle_gap = 2 * np.pi / len(angles) * 0.1
    for i in range(1, len(sorted_angles)):
        if sorted_angles[i] - sorted_angles[i-1] < min_angle_gap:
            sorted_angles[i] = sorted_angles[i-1] + min_angle_gap
    
    # Normalize angles to [0, 2*pi)
    sorted_angles = sorted_angles % (2 * np.pi)
    
    x = sorted_radii * np.cos(sorted_angles)
    y = sorted_radii * np.sin(sorted_angles)
    
    # Center at origin
    points = np.column_stack([x, y])
    points -= np.mean(points, axis=0)
    
    return points


def _objective_polar(params, temperature=0.05):
    """Objective function for polar parameterization.
    params: [r1, r2, ..., rN, a1, a2, ..., aN]
    Returns negative soft-min score (to minimize)."""
    radii = params[:N]
    angles = params[N:]
    
    points = _polar_to_cartesian(radii, angles)
    
    return _objective_softmin(points, temperature)


def _initial_regular_polygon(n):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _initial_concentric_rings(n, n_outer=None, n_inner=None):
    """Generate points in concentric rings."""
    if n_outer is None:
        n_outer = int(n * 0.7)
    n_inner = n - n_outer
    
    points = []
    
    # Outer ring
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer_radii = np.ones(n_outer)
    for i in range(n_outer):
        points.append([outer_radii[i] * np.cos(outer_angles[i]), 
                       outer_radii[i] * np.sin(outer_angles[i])])
    
    # Inner ring
    if n_inner > 0:
        inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        inner_radii = np.ones(n_inner) * 0.4
        for i in range(n_inner):
            points.append([inner_radii[i] * np.cos(inner_angles[i]), 
                           inner_radii[i] * np.sin(inner_angles[i])])
    
    return np.array(points)


def _lbfgs_polish(points, max_iter=500):
    """Polish points using L-BFGS-B optimizer."""
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return _objective_softmin(pts, temperature=0.03)
    
    try:
        result = minimize(
            obj, x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-10}
        )
        opt_points = result.x.reshape(-1, 2)
        return opt_points
    except Exception:
        return points


def _simulated_annealing(points, n_steps=5000, t0=0.1, tf=0.001, alpha=0.999):
    """Simulated annealing to escape local minima."""
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    for step in range(n_steps):
        temperature = t0 * (tf / t0) ** (step / n_steps)
        
        # Generate perturbation
        perturbation = rng.normal(0, temperature, size=current_points.shape)
        proposed_points = current_points + perturbation
        
        proposed_score = _score(proposed_points)
        
        # Accept if better or with probability exp((new-old)/T)
        if proposed_score > current_score:
            current_points = proposed_points
            current_score = proposed_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = proposed_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-10)):
                current_points = proposed_points
                current_score = proposed_score
    
    return best_points


def construct():
    """Construct 14 points to maximize minimum triangle area / convex hull area."""
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -1.0
    
    # Stage 1: Differential evolution in polar space
    # Parameters: [r1..r14, a1..a14]
    # Bounds: radii in [0.1, 2.0], angles in [0, 2*pi]
    
    bounds = [(0.1, 2.0)] * N + [(0, 2 * np.pi)] * N
    
    # Use differential evolution for global search
    de_result = differential_evolution(
        _objective_polar,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        tol=1e-6,
        polish=False,  # We'll do our own polishing
        init='latinhypercube',
        updating='immediate'
    )
    
    if de_result.success:
        de_params = de_result.x
        de_points = _polar_to_cartesian(de_params[:N], de_params[N:])
        de_score = _score(de_points)
        
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
    
    # Stage 2: Polish with L-BFGS-B
    if best_points is not None:
        polished = _lbfgs_polish(best_points, max_iter=1000)
        polished_score = _score(polished)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished.copy()
    
    # Stage 3: Try structured initializations and polish
    initializations = [
        _initial_regular_polygon(N),
        _initial_concentric_rings(N, 10, 4),
        _initial_concentric_rings(N, 8, 6),
        _initial_concentric_rings(N, 7, 7),
    ]
    
    for init_points in initializations:
        if time.time() - start_time > deadline - 15.0:
            break
        
        # Polish each initialization
        polished = _lbfgs_polish(init_points, max_iter=1500)
        polished_score = _score(polished)
        
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished.copy()
    
    # Stage 4: Simulated annealing from best point
    if best_points is not None and time.time() - start_time < deadline - 20.0:
        sa_points = _simulated_annealing(best_points, n_steps=3000, t0=0.05, tf=0.001)
        sa_score = _score(sa_points)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Stage 5: Final polish
    if best_points is not None:
        final_polished = _lbfgs_polish(best_points, max_iter=2000)
        final_score = _score(final_polished)
        
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished.copy()
    
    # If no valid result, return a default convex polygon
    if best_points is None:
        best_points = _initial_regular_polygon(N)
    
    return best_points

# EVOLVE-BLOCK-END
