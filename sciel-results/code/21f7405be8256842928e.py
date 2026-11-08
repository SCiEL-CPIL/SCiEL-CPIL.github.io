"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area of triangle using cross product
                area = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, 'volume' is the area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area. Higher is better."""
    min_tri = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_tri / hull_area


def _negative_score(params):
    """Objective function for differential_evolution (minimizes negative score)."""
    # params: [r0, a0, r1, a1, ..., r13, a13]
    radii = params[0::2]
    angles = params[1::2]
    
    # Enforce strictly increasing angles by sorting and wrapping
    # Actually, for polar coords on a disk, we can just use the angles as-is
    # but to avoid self-intersection issues in hull, let's sort by angle
    order = np.argsort(angles)
    sorted_angles = angles[order]
    sorted_radii = radii[order]
    
    # Convert to Cartesian
    points = np.column_stack([sorted_radii * np.cos(sorted_angles),
                              sorted_radii * np.sin(sorted_angles)])
    
    # Ensure points are within unit disk (they should be by bounds)
    # Normalize by convex hull area
    score = _score(points)
    return -score  # Minimize negative score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: 130s out of 150s
    
    # Initial good heuristic: points on a circle (regular 14-gon)
    angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii_init = np.ones(N)
    points_init = np.column_stack([radii_init * np.cos(angles_init),
                                   radii_init * np.sin(angles_init)])
    best_score = _score(points_init)
    best_points = points_init.copy()
    
    # Differential evolution in polar coordinates
    # Bounds: radius in [0.1, 1.0], angle in [0, 2*pi]
    # We'll optimize over unsorted angles and let the function sort them
    bounds = [(0.1, 1.0), (0, 2 * np.pi)] * N
    
    # Use a moderate population size and maxiter
    # We need to be careful with time
    try:
        def objective_with_deadline(params):
            if time.time() > deadline:
                raise TimeoutError()
            return _negative_score(params)
        
        result = differential_evolution(
            objective_with_deadline,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=N,
            polish=True
        )
        
        if result.success or result.fun < 0:
            # Convert result back to points
            radii = result.x[0::2]
            angles = result.x[1::2]
            order = np.argsort(angles)
            sorted_angles = angles[order]
            sorted_radii = radii[order]
            points_de = np.column_stack([sorted_radii * np.cos(sorted_angles),
                                         sorted_radii * np.sin(sorted_angles)])
            score_de = _score(points_de)
            if score_de > best_score:
                best_score = score_de
                best_points = points_de
    except Exception:
        pass
    
    # Additional local refinement if time permits
    if time.time() < deadline - 10:
        # Try a few random perturbations around best
        rng = np.random.default_rng(N)
        for _ in range(100):
            if time.time() > deadline:
                break
            perturb = best_points + rng.normal(0, 0.01, size=best_points.shape)
            # Keep within unit disk
            norms = np.linalg.norm(perturb, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                perturb[mask] /= norms[mask, np.newaxis] * 1.001
            score_pert = _score(perturb)
            if score_pert > best_score:
                best_score = score_pert
                best_points = perturb
    
    return best_points
# EVOLVE-BLOCK-END
