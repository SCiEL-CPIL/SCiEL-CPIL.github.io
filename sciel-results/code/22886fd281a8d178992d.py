# EVOLVE-BLOCK-START
import numpy as np
import time

def _vertex_score(distances):
    """
    distances: array of 19 distances from vertex i to others.
    Returns score for this vertex in [0, 1].
    """
    d = np.sort(distances)
    
    if np.any(d < 0.05):
        return 0.0
        
    diffs = d[1:] - d[:-1]
    k = np.argmin(diffs)
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
        
    if len(selected) != 4:
        return 0.0

    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _is_valid_convex(points):
    """Check if points form a strictly convex polygon."""
    n = len(points)
    
    # Check for repeated points
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-6:
        return False
        
    # Check turns
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    if np.any(np.abs(cross) < 1e-8):
        return False
        
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
        
    return True

def _polygon_score(points):
    """
    points: (20, 2) array.
    Returns minimum vertex score after normalization.
    """
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    
    dists_to_c = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_c)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points_norm = points_norm / mean_dist
    
    if not _is_valid_convex(points_norm):
        return 0.0
        
    n = len(points_norm)
    min_score = 1.0
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        s = _vertex_score(dists)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
            
    return min_score

def _compute_all_scores(points):
    """
    Compute normalized points and all vertex scores.
    Returns (points_norm, scores, valid)
    """
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    
    dists_to_c = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_c)
    
    if mean_dist <= 0.01:
        return None, None, False
        
    points_norm = points_norm / mean_dist
    
    if not _is_valid_convex(points_norm):
        return None, None, False
        
    n = len(points_norm)
    scores = np.zeros(n)
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        scores[i] = _vertex_score(dists)
        
    return points_norm, scores, True

def _soft_min(scores, sharpness=30.0):
    """
    Compute soft minimum to provide smooth gradient for optimization.
    """
    return -np.log(np.sum(np.exp(-sharpness * scores))) / sharpness

def _smooth_objective(points):
    """
    Smooth objective function for coordinate descent.
    Returns negative soft-min score (lower is better).
    """
    _, scores, valid = _compute_all_scores(points)
    if not valid:
        return 100.0
    return -_soft_min(scores, sharpness=30.0)

def _make_regular_polygon(n=20, radius=1.0):
    """Create a regular polygon."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)

def _coordinate_descent(points, max_iters=500, step_size=0.02, deadline=None):
    """
    Coordinate descent on individual coordinates, maintaining convexity.
    """
    current = points.copy()
    current_obj = _smooth_objective(current)
    n = len(current)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for coord in range(2):
                # Try small perturbations
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    trial = current.copy()
                    trial[i, coord] += delta
                    trial_obj = _smooth_objective(trial)
                    if trial_obj < current_obj - 1e-8:
                        current = trial
                        current_obj = trial_obj
                        improved = True
                        break
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return current

def _simulated_annealing(points, max_iters=3000, deadline=None):
    """
    Simulated annealing on the polygon coordinates.
    """
    current = points.copy()
    current_obj = _smooth_objective(current)
    n = len(current)
    
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    
    best = current.copy()
    best_obj = current_obj
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        i = np.random.randint(n)
        coord = np.random.randint(2)
        delta = np.random.normal(0, 0.05)
        
        trial = current.copy()
        trial[i, coord] += delta
        
        trial_obj = _smooth_objective(trial)
        
        # Acceptance criterion
        if trial_obj < current_obj:
            current = trial
            current_obj = trial_obj
            if current_obj < best_obj:
                best = current.copy()
                best_obj = current_obj
        else:
            # Accept with probability exp(-(delta_obj) / T)
            delta_obj = trial_obj - current_obj
            if np.random.random() < np.exp(-delta_obj / T):
                current = trial
                current_obj = trial_obj
        
        T = max(T_min, T * cooling)
    
    return best

def _harmonic_perturbation(points, n_harmonics=5, amplitude=0.1, deadline=None):
    """
    Add harmonic perturbations to a regular polygon and optimize.
    This exploits the near-equidistance of regular polygons.
    """
    n = len(points)
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_points = points.copy()
    best_score = _polygon_score(points)
    
    for harmonic in range(1, n_harmonics + 1):
        for phase in np.linspace(0, 2*np.pi, 8, endpoint=False):
            if deadline is not None and time.time() > deadline:
                return best_points
            
            # Create perturbed polygon
            radii = 1.0 + amplitude * np.sin(harmonic * base_angles + phase)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            trial = np.stack([x, y], axis=1)
            
            # Refine with coordinate descent
            trial = _coordinate_descent(trial, max_iters=100, step_size=0.01, deadline=deadline)
            
            score = _polygon_score(trial)
            if score > best_score:
                best_score = score
                best_points = trial.copy()
    
    return best_points

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    This baseline is the upstream search's starting construction: a regular polygon whose
    vertices lie evenly on a circle (deterministic, strictly convex). The scoring function
    is scale/translation invariant, so the circle radius is arbitrary.

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    n = 20
    
    # Start with regular polygon
    best_points = _make_regular_polygon(n)
    best_score = _polygon_score(best_points)
    
    # Strategy 1: Harmonic perturbations + coordinate descent
    # This is structural: it explores a specific family of near-regular polygons
    # that may have better equidistance properties
    harmonic_results = _harmonic_perturbation(
        best_points, 
        n_harmonics=5, 
        amplitude=0.1, 
        deadline=deadline
    )
    harmonic_score = _polygon_score(harmonic_results)
    if harmonic_score > best_score:
        best_score = harmonic_score
        best_points = harmonic_results
    
    # Strategy 2: Simulated annealing from the best point found
    if time.time() < deadline - 60:
        sa_results = _simulated_annealing(
            best_points, 
            max_iters=2000, 
            deadline=deadline - 30
        )
        sa_score = _polygon_score(sa_results)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_results
    
    # Strategy 3: Final coordinate descent refinement
    if time.time() < deadline - 10:
        final_results = _coordinate_descent(
            best_points, 
            max_iters=300, 
            step_size=0.005,
            deadline=deadline
        )
        final_score = _polygon_score(final_results)
        if final_score > best_score:
            best_score = final_score
            best_points = final_results
    
    return best_points

# EVOLVE-BLOCK-END
