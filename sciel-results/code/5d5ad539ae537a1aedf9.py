import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        # Check for collinearity or duplicates which cause errors
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectors from i to j and i to k
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            # Cross product magnitude / 2
            cross = dx_ji * dy_ki - dy_ji * dx_ki
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

def _initial_regular_polygon(n, noise=0.0):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    if noise > 0:
        rng = np.random.RandomState(np.random.randint(10000))
        pts = pts + rng.normal(0, noise, size=pts.shape)
    # Normalize to unit area
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts = pts / np.sqrt(hull_a)
    return pts

def _initial_mixed(n):
    """Mix of boundary and interior points."""
    outer_n = 10
    inner_n = n - outer_n
    angles_outer = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer, inner])
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts = pts / np.sqrt(hull_a)
    return pts

def _initial_random(n, seed):
    """Random points in a circle, scaled."""
    rng = np.random.RandomState(seed)
    r = rng.uniform(0.1, 1.0, size=n)
    theta = rng.uniform(0, 2*np.pi, size=n)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    hull_a = _hull_area(pts)
    if hull_a > 0:
        pts = pts / np.sqrt(hull_a)
    return pts

def _normalize_unit_area(pts):
    """Scale points so that their convex hull has area 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts

def _simulated_annealing(pts, start_time, deadline, rng):
    """
    Perform Simulated Annealing to maximize the score.
    Uses a cooling schedule and accepts worse moves with probability exp(-delta/T).
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T_start = 0.1
    T_end = 1e-6
    total_steps = 5000 # Estimate steps based on time budget
    
    # Pre-calculate step size decay
    # We want to move roughly O(1) in the beginning and O(1e-4) at the end
    step_start = 0.5
    step_end = 0.001
    
    for step in range(total_steps):
        if time.time() > deadline:
            break
            
        # Calculate current temperature and step size
        progress = step / total_steps
        T = T_start * (T_end / T_start) ** progress
        step_size = step_start * (step_end / step_start) ** progress
        
        # Generate a new candidate point
        # Perturb a random subset of points to allow for local rearrangements
        # or perturb all points slightly for global shape changes
        if rng.random() < 0.1:
            # Global move: perturb all points
            noise = rng.normal(0, step_size, size=current_pts.shape)
        else:
            # Local move: perturb 2-4 random points
            noise = np.zeros_like(current_pts)
            num_moved = rng.randint(2, 5)
            indices = rng.choice(N, num_moved, replace=False)
            noise[indices] = rng.normal(0, step_size * 2, size=(num_moved, 2))
            
        candidate_pts = current_pts + noise
        
        # Normalize to maintain scale invariance (hull area = 1)
        candidate_pts = _normalize_unit_area(candidate_pts)
        
        # Evaluate
        candidate_score = _score(candidate_pts)
        
        # Acceptance criterion
        if candidate_score > current_score:
            current_pts = candidate_pts
            current_score = candidate_score
            if current_score > best_score:
                best_pts = current_pts.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            delta = current_score - candidate_score
            if delta < 1e-12:
                continue
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_pts = candidate_pts
                current_score = candidate_score
                
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Portfolio of initializations
    candidates = []
    
    # 1. Regular polygon variants
    for noise in [0.0, 0.01, 0.05, 0.1]:
        candidates.append(_initial_regular_polygon(N, noise))
        
    # 2. Mixed boundary/interior
    candidates.append(_initial_mixed(N))
    
    # 3. Random initializations (different seeds)
    for seed in range(10):
        candidates.append(_initial_random(N, seed))
        
    # Evaluate initial scores
    scores = np.array([_score(c) for c in candidates])
    
    # Sort candidates by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    candidates = [candidates[i] for i in sorted_indices]
    scores = scores[sorted_indices]
    
    best_pts = candidates[0]
    best_score = scores[0]
    
    # Simulated Annealing on top candidates
    # We allocate time budget to the top few candidates
    top_k = 5
    time_per_candidate = (deadline - time.time()) / top_k - 2.0 # Leave 2s for final polish
    
    for i in range(top_k):
        if time.time() > deadline - 2.0:
            break
            
        # Use a different random seed for each candidate to diversify search
        rng = np.random.RandomState(12345 + i)
        
        # Adjust steps based on available time for this candidate
        # Estimate time per step is roughly constant
        # We'll just run SA until time limit for this candidate slice
        cand_start = time.time()
        cand_deadline = min(deadline - 2.0, cand_start + time_per_candidate)
        
        improved_pts, improved_score = _simulated_annealing(candidates[i], cand_start, cand_deadline, rng)
        
        if improved_score > best_score:
            best_score = improved_score
            best_pts = improved_pts
            
    # Final polish: Run a short, low-temperature SA on the global best
    if time.time() < deadline - 1.0:
        rng = np.random.RandomState(99999)
        final_pts, final_score = _simulated_annealing(best_pts, time.time(), deadline - 0.5, rng)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    return best_pts

# EVOLVE-BLOCK-END
