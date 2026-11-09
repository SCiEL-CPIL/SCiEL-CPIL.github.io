"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    hull = ConvexHull(points)
    return hull.volume


def _score_from_angles(angles):
    """
    Given central angles for N points on a unit circle,
    compute the score: min_triangle_area / convex_hull_area.
    """
    angles = np.sort(angles) % (2 * np.pi)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    
    if hull_area <= 0:
        return 0.0
    
    return min_area / hull_area


def _enforce_separation(angles, min_sep=0.05):
    """Ensure minimum angular separation between consecutive points."""
    angles = np.sort(angles) % (2 * np.pi)
    n = len(angles)
    
    # Check and fix separations
    for _ in range(10):
        gaps = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
        min_gap = np.min(gaps)
        if min_gap >= min_sep:
            break
        # Find the smallest gap and spread it
        min_idx = np.argmin(gaps)
        shift = (min_sep - min_gap) / 2.0
        angles[min_idx:] += shift
        angles = angles % (2 * np.pi)
        angles = np.sort(angles)
    
    return angles


def _hill_climb_optimize(angles, iterations=5000, step_init=0.1, step_min=0.001):
    """
    Custom gradient-free hill climbing with adaptive step size.
    Optimizes angles to maximize the score.
    """
    best_angles = _enforce_separation(angles.copy())
    best_score = _score_from_angles(best_angles)
    
    step_size = step_init
    no_improve = 0
    max_no_improve = 100
    
    for it in range(iterations):
        if no_improve > max_no_improve:
            break
        
        # Try random perturbations
        improved = False
        for _ in range(10):
            trial = best_angles.copy()
            idx = np.random.randint(N)
            trial[idx] += np.random.uniform(-step_size, step_size)
            trial = _enforce_separation(trial)
            
            trial_score = _score_from_angles(trial)
            if trial_score > best_score + 1e-10:
                best_angles = trial
                best_score = trial_score
                improved = True
                no_improve = 0
                break
        else:
            no_improve += 1
            step_size = max(step_min, step_size * 0.95)
    
    return best_angles, best_score


def _multi_start_optimize(n_starts=5, seed=42):
    """
    Multi-start optimization with diverse initial configurations.
    """
    rng = np.random.RandomState(seed)
    
    best_overall = None
    best_overall_score = -np.inf
    
    for start in range(n_starts):
        if start == 0:
            # Evenly spaced initial guess
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        elif start == 1:
            # Slightly perturbed even spacing
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            angles += rng.uniform(-0.1, 0.1, N)
        else:
            # Random initial angles
            angles = rng.uniform(0, 2 * np.pi, N)
        
        angles = _enforce_separation(angles)
        
        # Run hill climbing
        opt_angles, opt_score = _hill_climb_optimize(
            angles,
            iterations=3000,
            step_init=0.15,
            step_min=0.002
        )
        
        if opt_score > best_overall_score:
            best_overall = opt_angles
            best_overall_score = opt_score
    
    return best_overall, best_overall_score


def construct():
    """
    Place 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Approach: Represent points as inscribed in a unit circle, parameterized
    by central angles. Use a custom gradient-free hill climbing algorithm
    with angular separation constraints and multi-start strategy.
    """
    best_angles, best_score = _multi_start_optimize(n_starts=8, seed=42)
    
    # Convert to points on unit circle
    points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return points
# EVOLVE-BLOCK-END
