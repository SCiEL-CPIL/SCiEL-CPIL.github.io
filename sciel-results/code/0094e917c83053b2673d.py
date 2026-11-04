"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Geometrically-aware Simulated Annealing with Worst-Triangle Repair 
and Nelder-Mead local polishing on a soft-minimum surrogate.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_neldermead

N = 13

# EVOLVE-BLOCK-START

def _compute_score_and_worst(pts):
    """
    Calculate the score and identify the worst triangle.
    Returns (score, worst_indices) where worst_indices is a tuple (i, j, k) or None.
    """
    try:
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

def _soft_min_objective(x, sharpness=50.0):
    """
    Differentiable objective function for optimization.
    Minimizes -soft_min(areas) / hull_area.
    """
    pts = x.reshape(N, 2)
    
    # Compute areas
    i, j, k = np.indices((N, N, N))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min
    beta = sharpness
    max_neg_area = np.max(-areas)
    log_sum_exp = max_neg_area + np.log(np.sum(np.exp(-areas - max_neg_area)))
    soft_min_val = -log_sum_exp / beta
    
    # Hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10 # Large penalty
    
    if hull_area < 1e-10:
        return 1e10
        
    score = soft_min_val / hull_area
    return -score

def _local_polish(pts, time_limit=2.0):
    """
    Apply Nelder-Mead to the soft-min objective starting from pts.
    Returns improved points and score.
    """
    x0 = pts.flatten()
    start_time = time.time()
    
    # Use a moderate sharpness for smoothness
    sharpness = 50.0
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10 # Force termination
        return _soft_min_objective(x, sharpness)
    
    try:
        res = minimize_neldermead(objective, x0, 
                                   options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
        
        # Check if we improved
        new_pts = res.x.reshape(N, 2)
        new_score, _ = _compute_score_and_worst(new_pts)
        old_score, _ = _compute_score_and_worst(pts)
        
        if new_score > old_score:
            return new_pts, new_score
        else:
            return pts, old_score
    except Exception:
        return pts, _compute_score_and_worst(pts)[0]

def _worst_triangle_repair(pts, worst_idx, strength=0.1):
    """
    Attempt to fix the worst triangle by moving one of its vertices 
    away from the other two, while maintaining convexity roughly.
    """
    if worst_idx is None:
        return pts
    
    i, j, k = worst_idx
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    # Calculate centroid of the worst triangle
    centroid = (p_i + p_j + p_k) / 3.0
    
    # We want to move one point away. 
    # Let's pick the point that is "most central" or just randomize.
    # A good heuristic: move the point that is closest to the centroid of the *other* points?
    # Or just move each vertex away from the centroid of the triangle.
    
    # Let's try moving point i away from the line segment jk.
    # Vector jk
    v_jk = p_k - p_j
    # Normal to jk
    if np.linalg.norm(v_jk) < 1e-9:
        return pts
    n_jk = np.array([-v_jk[1], v_jk[0]])
    n_jk /= np.linalg.norm(n_jk)
    
    # Determine which side of the line jk point i is on
    cross_check = np.cross(v_jk, p_i - p_j)
    if cross_check > 0:
        direction = n_jk
    else:
        direction = -n_jk
        
    # Move i in that direction
    new_pts = pts.copy()
    new_pts[i] = p_i + direction * strength * np.linalg.norm(v_jk)
    
    # Also try moving j away from ik
    v_ik = p_k - p_i
    if np.linalg.norm(v_ik) > 1e-9:
        n_ik = np.array([-v_ik[1], v_ik[0]])
        n_ik /= np.linalg.norm(n_ik)
        cross_check_j = np.cross(v_ik, p_j - p_i)
        if cross_check_j > 0:
            dir_j = n_ik
        else:
            dir_j = -n_ik
        new_pts[j] = p_j + dir_j * strength * np.linalg.norm(v_ik)
        
    # Also try moving k away from ij
    v_ij = p_j - p_i
    if np.linalg.norm(v_ij) > 1e-9:
        n_ij = np.array([-v_ij[1], v_ij[0]])
        n_ij /= np.linalg.norm(n_ij)
        cross_check_k = np.cross(v_ij, p_k - p_i)
        if cross_check_k > 0:
            dir_k = n_ij
        else:
            dir_k = -n_ij
        new_pts[k] = p_k + dir_k * strength * np.linalg.norm(v_ij)
        
    return new_pts

def _simulated_annealing(pts, time_limit=30.0, rng=None):
    """
    Run Simulated Annealing.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    current_pts = pts.copy()
    current_score, current_worst = _compute_score_and_worst(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 0.5 # Initial temperature (scale of moves)
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Generate a move
        # 1. Random perturbation
        move_type = rng.random()
        
        if move_type < 0.7:
            # Small random jitter
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, 2)
            new_pts = current_pts.copy()
            new_pts[idx] += delta
        elif move_type < 0.85:
            # Worst triangle repair
            new_pts = _worst_triangle_repair(current_pts, current_worst, strength=T)
        else:
            # Rotate a pair of points around their midpoint (preserves centroid locally)
            i, j = rng.choice(N, 2, replace=False)
            mid = (current_pts[i] + current_pts[j]) / 2.0
            v = current_pts[j] - current_pts[i]
            angle = rng.normal(0, T)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            new_pts = current_pts.copy()
            new_pts[i] = mid - R @ v / 2.0
            new_pts[j] = mid + R @ v / 2.0
            
        new_score, new_worst = _compute_score_and_worst(new_pts)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            current_pts = new_pts
            current_score = new_score
            current_worst = new_worst
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
                
        T *= cooling
        
        # Occasionally polish
        if rng.random() < 0.05:
            polished_pts, polished_score = _local_polish(current_pts, time_limit=1.0)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
                current_pts = polished_pts
                current_score = polished_score
                # Recalculate worst for current
                _, current_worst = _compute_score_and_worst(current_pts)

    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initial points
    # Use Halton sequences for quasi-random distribution
    def halton_sequence(n, base):
        seq = np.zeros((n, 2))
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += rem * f
            seq[i-1, 0] = r
            # For y coordinate, use next base or different logic? 
            # Standard 2D Halton uses base 2 for x, base 3 for y
            f = 1.0
            r = 0.0
            i_val = i
            base_y = 3 if base == 2 else 2
            while i_val > 0:
                i_val, rem = divmod(i_val, base_y)
                f /= base
                r += rem * f
            seq[i-1, 1] = r
        return seq

    starts = []
    
    # Halton 2
    pts_h = halton_sequence(N, 2)
    try:
        hull = ConvexHull(pts_h)
        area = hull.volume
        if area > 1e-10:
            pts_h /= np.sqrt(area)
    except:
        pass
    starts.append(pts_h)
    
    # Halton 3
    pts_h3 = halton_sequence(N, 3)
    try:
        hull = ConvexHull(pts_h3)
        area = hull.volume
        if area > 1e-10:
            pts_h3 /= np.sqrt(area)
    except:
        pass
    starts.append(pts_h3)
    
    # Random
    pts_r = rng.normal(0, 1, (N, 2))
    try:
        hull = ConvexHull(pts_r)
        area = hull.volume
        if area > 1e-10:
            pts_r /= np.sqrt(area)
    except:
        pass
    starts.append(pts_r)
    
    # Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_p = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(pts_p)
    
    best_pts = None
    best_score = -1.0
    
    # 2. Run Simulated Annealing from each start
    # We have ~100s budget. Let's allocate ~20s per start.
    time_per_start = 20.0
    
    for s in starts:
        if time.time() > deadline - 10:
            break
            
        # Normalize start to roughly unit area
        try:
            hull = ConvexHull(s)
            area = hull.volume
            if area > 1e-10:
                s = s / np.sqrt(area)
        except:
            continue
            
        sa_pts, sa_score = _simulated_annealing(s, time_limit=time_per_start, rng=rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    # 3. Final Polish on the best result
    if best_pts is not None:
        polished_pts, polished_score = _local_polish(best_pts, time_limit=10.0)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
            
    # 4. Fallback
    if best_pts is None or best_score < 1e-6:
        # Absolute fallback
        best_pts = starts[0]
        best_score, _ = _compute_score_and_worst(best_pts)

    return best_pts

# EVOLVE-BLOCK-END
