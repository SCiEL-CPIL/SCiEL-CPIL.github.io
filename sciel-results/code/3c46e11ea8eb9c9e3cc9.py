"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by triples of points.
    Uses vectorization for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized calculation of all triangle areas
    # We can compute cross products for all triples i < j < k
    # To avoid O(N^3) memory blowup for large N, we can do it in chunks or just rely on N=14 being small enough for full broadcast.
    # For N=14, number of triangles is 364. This is very small.
    # Let's just compute all pairwise cross products relative to each point? No, simpler:
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # Since N is small (14), we can just use a loop for i and vectorize j, k
    min_area = np.inf
    for i in range(n - 2):
        pi = points[i]
        # Vectors from pi to all other points
        # We only need j > i and k > j
        # Let's just compute for all j, k > i
        remaining = points[i+1:]
        m = len(remaining)
        if m < 2:
            continue
        
        # Compute cross product of vectors (pj - pi) and (pk - pi) for all j, k in remaining
        # remaining is shape (m, 2)
        # We want cross(u, v) for u, v in remaining
        # cross(u, v) = u[0]*v[1] - u[1]*v[0]
        # u = remaining[j], v = remaining[k]
        # We can use broadcasting: remaining[:, 0][None, :] * remaining[1:, 1] - ... 
        # Actually, let's just compute the matrix of cross products for all pairs in 'remaining'
        # But we only care about j < k in the original indexing, which corresponds to all pairs in 'remaining'
        
        # Let u = remaining[:, 0], v = remaining[:, 1]
        # Cross product matrix C where C[a, b] = cross(remaining[a], remaining[b])
        # C[a, b] = u[a]*v[b] - v[a]*u[b]
        u = remaining[:, 0]
        v = remaining[:, 1]
        
        # C = np.outer(u, v) - np.outer(v, u)
        # This gives an m x m matrix.
        # We want min over a < b of 0.5 * abs(C[a, b])
        # Since C is antisymmetric, C[a, b] = -C[b, a].
        # We can just take the upper triangle.
        
        C = np.outer(u, v) - np.outer(v, u)
        # Mask for upper triangle
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        areas = 0.5 * np.abs(C[mask])
        if len(areas) > 0:
            min_area = min(min_area, np.min(areas))
            
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    hull = ConvexHull(points)
    return hull.volume


def _score_from_angles(angles):
    """
    Given central angles for N points on a unit circle,
    compute the score: min_triangle_area / convex_hull_area.
    Note: For points on a unit circle, the convex hull is the regular polygon if points are sorted and cover the circle,
    but generally it's the polygon formed by the points.
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


def _hill_climb_optimize(angles, iterations=2000, step_init=0.1, step_min=0.0001, rng=None):
    """
    Improved gradient-free hill climbing.
    """
    if rng is None:
        rng = np.random.RandomState(0)
        
    best_angles = angles.copy()
    best_score = _score_from_angles(best_angles)
    
    step_size = step_init
    no_improve = 0
    max_no_improve = 50
    
    for it in range(iterations):
        if no_improve > max_no_improve:
            break
        
        improved = False
        # Try multiple perturbations per iteration to increase chance of improvement
        for _ in range(20):
            trial = best_angles.copy()
            # Perturb 1 or 2 angles
            n_perturb = 1 if rng.rand() < 0.8 else 2
            indices = rng.choice(N, size=n_perturb, replace=False)
            for idx in indices:
                trial[idx] += rng.uniform(-step_size, step_size)
            
            # Enforce separation softly or just normalize
            trial = np.sort(trial) % (2 * np.pi)
            
            trial_score = _score_from_angles(trial)
            if trial_score > best_score + 1e-12:
                best_angles = trial
                best_score = trial_score
                improved = True
                no_improve = 0
                break
        else:
            no_improve += 1
            # Decay step size
            step_size = max(step_min, step_size * 0.9)
    
    return best_angles, best_score


def _multi_start_optimize(n_starts=10, seed=42):
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
            angles += rng.uniform(-0.05, 0.05, N)
        elif start == 2:
            # Another perturbation
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            angles += rng.uniform(-0.2, 0.2, N)
        else:
            # Random initial angles
            angles = rng.uniform(0, 2 * np.pi, N)
        
        angles = np.sort(angles) % (2 * np.pi)
        
        # Run hill climbing
        opt_angles, opt_score = _hill_climb_optimize(
            angles,
            iterations=1000,
            step_init=0.1,
            step_min=0.0005,
            rng=rng
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
    # Use a time limit to ensure we finish within 120s
    import time
    start_time = time.time()
    
    # Run multiple seeds to ensure robustness
    best_angles = None
    best_score = -np.inf
    
    for seed in [42, 123, 999, 7, 55]:
        if time.time() - start_time > 100: # Leave 20s buffer
            break
        angles, score = _multi_start_optimize(n_starts=5, seed=seed)
        if score > best_score:
            best_score = score
            best_angles = angles
    
    # Final polish with more iterations on the best found
    if best_angles is not None:
        # One more round of refinement
        final_angles, final_score = _hill_climb_optimize(
            best_angles,
            iterations=2000,
            step_init=0.05,
            step_min=0.0001,
            rng=np.random.RandomState(42)
        )
        if final_score > best_score:
            best_angles = final_angles
            best_score = final_score
    
    # Convert to points on unit circle
    points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return points
# EVOLVE-BLOCK-END
