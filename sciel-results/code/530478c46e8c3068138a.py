"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import random

N = 13

# EVOLVE-BLOCK-START
def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    # Precompute all cross products for triangles
    # Use vectorized approach: for each pair (i,j), compute cross with all k
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized: cross product of (pts[j]-pts[i]) with (pts[k]-pts[i]) for all k
            v1 = pts[j] - pts[i]
            v2 = pts - pts[i]
            # Cross product in 2D: v1_x * v2_y - v1_y * v2_x
            crosses = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
            # Area is 0.5 * |cross|, but we can work with |cross| and divide at end
            # Exclude i and j themselves (cross will be 0 anyway)
            abs_crosses = np.abs(crosses)
            # The minimum non-zero cross for this pair
            # Actually we want the global minimum over all triples
            # For this pair (i,j), the minimum area triangle involving i,j is min over k != i,j
            # But we need to exclude k=i and k=j
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            if np.any(mask):
                min_cross = np.min(abs_crosses[mask])
                if min_cross < min_area:
                    min_area = min_cross
    return 0.5 * min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _fermat_spiral(n):
    """Generate n points using Fermat/Vogel spiral."""
    i = np.arange(n)
    r = np.sqrt((i + 0.5) / n)
    theta = (i + 0.5) * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    pts -= np.mean(pts, axis=0)
    return pts


def _simulated_annealing(pts0, max_time, rng):
    """Simulated annealing to maximize the minimum triangle area ratio."""
    n = len(pts0)
    pts = pts0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initial temperature
    T = 0.1
    cooling = 0.9995
    step_scale = 0.1
    
    while time.time() < deadline:
        # Generate a random perturbation
        idx = rng.randint(0, n - 1)
        delta = np.array([rng.gauss(0, step_scale), rng.gauss(0, step_scale)])
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        
        new_score = _score(new_pts)
        old_score = _score(pts)
        
        if new_score >= old_score or rng.random() < np.exp((new_score - old_score) / T):
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        
        T *= cooling
        step_scale *= 0.999
        
        # Check deadline more frequently
        if time.time() > deadline - 5:
            break
    
    return best_pts, best_score


def _refine_with_gradient(pts, max_time, rng):
    """Local refinement using coordinate descent on the worst triangle."""
    n = len(pts)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    start_time = time.time()
    deadline = start_time + max_time
    
    step_size = 0.05
    
    while time.time() < deadline:
        score = _score(best_pts)
        
        # Find the worst triangle
        min_area = np.inf
        worst_tri = (0, 1, 2)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    v1 = best_pts[j] - best_pts[i]
                    v2 = best_pts[k] - best_pts[i]
                    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if cross < min_area:
                        min_area = cross
                        worst_tri = (i, j, k)
        
        # Try to improve by moving one of the worst triangle vertices
        i, j, k = worst_tri
        for idx in [i, j, k]:
            # Try small moves in random directions
            for _ in range(10):
                direction = np.array([rng.gauss(0, 1), rng.gauss(0, 1)])
                direction /= (np.linalg.norm(direction) + 1e-12)
                
                for sign in [1, -1]:
                    new_pts = best_pts.copy()
                    new_pts[idx] += sign * step_size * direction
                    new_score = _score(new_pts)
                    if new_score > score:
                        best_pts = new_pts
                        score = new_score
                        break
        
        # Decay step size
        step_size *= 0.95
        if step_size < 1e-6:
            break
    
    return best_pts, best_score


def construct():
    """Construct 13 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    total_budget = 140  # Leave 10s safety margin for 150s limit
    
    rng = random.Random(N)
    np_rng = np.random.RandomState(N)
    
    best_pts = None
    best_score = -np.inf
    
    # Phase 1: Multiple Fermat spiral starts with different rotations
    n_starts = 5
    time_per_start = total_budget / (n_starts + 3)  # Reserve time for later phases
    
    for start_idx in range(n_starts):
        if time.time() - start_time > total_budget - 20:
            break
        
        # Create a rotated/scaled Fermat spiral
        base = _fermat_spiral(N)
        
        # Apply random rotation
        angle = rng.random() * 2 * np.pi
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        pts = base @ rot.T
        
        # Small random perturbation
        pts += np_rng.normal(0, 0.01, size=(N, 2))
        
        # Simulated annealing
        remaining = total_budget - (time.time() - start_time)
        sa_time = min(time_per_start, remaining * 0.4)
        if sa_time < 1:
            sa_time = 1
        
        pts, score = _simulated_annealing(pts, sa_time, rng)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Local refinement
        remaining = total_budget - (time.time() - start_time)
        refine_time = min(time_per_start * 0.5, remaining * 0.2)
        if refine_time < 0.5:
            refine_time = 0.5
        
        pts, score = _refine_with_gradient(pts, refine_time, rng)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 2: Try a few random starts with SA
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5 and best_pts is not None:
        n_random_starts = min(3, int(remaining / 5))
        for _ in range(n_random_starts):
            if time.time() - start_time > total_budget - 5:
                break
            
            # Random points in unit disk
            r = np.sqrt(np_rng.random(N))
            theta = 2 * np.pi * np_rng.random(N)
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            sa_time = min(3, (total_budget - (time.time() - start_time)) * 0.3)
            if sa_time < 0.5:
                break
            
            pts, score = _simulated_annealing(pts, sa_time, rng)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: return Fermat spiral
        best_pts = _fermat_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
