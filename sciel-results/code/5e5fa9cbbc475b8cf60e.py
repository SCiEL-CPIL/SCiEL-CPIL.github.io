"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Discrete worst-triangle-driven local search (hill climbing).
Unlike the smooth gradient-based approach, this method identifies the specific
triple of points forming the minimum area triangle and performs targeted
convex combination moves with neighboring points in the angular ordering.
This directly addresses the bottleneck (the worst triangle) and avoids getting
stuck in local minima of the smooth surrogate that might not correspond to
the true discrete minimum.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _compute_score_and_worst(pts):
    """
    Calculate the score and identify the worst triangle.
    Returns (score, worst_indices) where worst_indices is a tuple (i, j, k) or None.
    """
    try:
        if np.allclose(pts, pts[0]):
            return 0.0, None
            
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0, None
        
    if hull_area < 1e-10:
        return 0.0, None

    n = len(pts)
    min_area = float('inf')
    worst_idx = None
    
    # Vectorized computation for speed
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    worst_idx = (i[min_idx], j[min_idx], k[min_idx])
    
    score = min_area / hull_area
    return score, worst_idx

def _compute_score(pts):
    """
    Wrapper for score only.
    """
    score, _ = _compute_score_and_worst(pts)
    return score

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0,1]^2 for a given base."""
    seq = np.zeros((n, 2))
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        for d in range(1, 3):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += rem * f
            seq[i-1, d-1] = r
    return seq

def _normalize_hull_unit_area(pts):
    """Normalize points so convex hull area is 1."""
    try:
        hull = ConvexHull(pts)
        area = hull.volume
        if area > 1e-10:
            pts = pts / np.sqrt(area)
    except Exception:
        pass
    return pts

def _local_search(pts, max_iters=200, time_limit=10.0):
    """
    Perform local search by targeting the worst triangle.
    For each point in the worst triangle, try moving it towards its angular neighbors.
    """
    start_time = time.time()
    current_score, worst_idx = _compute_score_and_worst(pts)
    
    if current_score == 0.0 or worst_idx is None:
        return pts

    n = len(pts)
    # Sort points by angle around centroid to maintain convex ordering
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sort_idx = np.argsort(angles)
    pts = pts[sort_idx]
    
    # Re-map worst indices to sorted order? 
    # Actually, easier to just work in the sorted order and recompute worst each time.
    current_score, worst_idx = _compute_score_and_worst(pts)
    
    if current_score == 0.0 or worst_idx is None:
        return pts

    best_pts = pts.copy()
    best_score = current_score
    
    # Define neighbor indices (circular)
    neighbors = [((i-1)%n, (i+1)%n) for i in range(n)]
    
    for it in range(max_iters):
        if time.time() > start_time + time_limit:
            break
            
        score, worst_idx = _compute_score_and_worst(best_pts)
        if score == 0.0 or worst_idx is None:
            break
            
        if score > best_score:
            best_score = score
            # Reset some counter or continue
            
        i, j, k = worst_idx
        
        # Try moving each point in the worst triangle towards its neighbors
        improved = False
        
        for p_idx in [i, j, k]:
            p_pos = p_idx
            p_neighbor_indices = neighbors[p_pos]
            
            for n_idx in p_neighbor_indices:
                # Try moving p_pos towards n_idx
                # alpha in [0.1, 0.9]
                for alpha in [0.2, 0.4, 0.6, 0.8]:
                    new_pts = best_pts.copy()
                    # Convex combination: new_p = (1-alpha)*p + alpha*n
                    new_pts[p_pos] = (1 - alpha) * best_pts[p_pos] + alpha * best_pts[n_idx]
                    
                    new_score, _ = _compute_score_and_worst(new_pts)
                    
                    if new_score > best_score + 1e-10:
                        best_pts = new_pts
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            # If no single point move improves, try small random perturbations of the worst points
            # to escape local minima
            rng = np.random.default_rng()
            for _ in range(5):
                p_idx = rng.choice([i, j, k])
                new_pts = best_pts.copy()
                # Small random step
                new_pts[p_idx] += rng.normal(0, 0.05, 2)
                new_score, _ = _compute_score_and_worst(new_pts)
                if new_score > best_score + 1e-10:
                    best_pts = new_pts
                    best_score = new_score
                    improved = True
                    break
            
            if not improved:
                break
                
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Multi-start local search
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate initial points
        if i == 0:
            # Start with Halton
            pts = _halton_sequence(N, 2)
        else:
            # Start with random points, slightly perturbed from regular polygon
            rng = np.random.default_rng(1000 + i)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.2, N)
            radii = 1.0 + rng.normal(0, 0.1, N)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
        # Normalize to unit area
        pts = _normalize_hull_unit_area(pts)
        
        # Run local search
        # Time limit per start: distribute remaining time
        remaining_time = deadline - time.time() - 5.0
        time_per_start = max(1.0, remaining_time / (num_starts - i))
        
        pts_opt = _local_search(pts, max_iters=100, time_limit=time_per_start)
        
        score_opt, _ = _compute_score_and_worst(pts_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    if best_pts is None:
        # Fallback
        pts = _halton_sequence(N, 2)
        best_pts = _normalize_hull_unit_area(pts)

    return best_pts

# EVOLVE-BLOCK-END
