import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 21
DEADLINE_MARGIN = 10.0

def _get_candidates(w, h, grid_res=30):
    """Generate a grid of candidate positions."""
    # Add margin to ensure points are not too close to edge for radius calculation?
    # Actually, we want points near edges too.
    xs = np.linspace(0, w, grid_res)
    ys = np.linspace(0, h, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points

def _calc_radii(points, w, h):
    """Calculate max radius for each point given the set of points."""
    n = len(points)
    if n == 0: return np.array([])
    
    # Distance to walls
    d_left = points[:, 0]
    d_right = w - points[:, 0]
    d_bottom = points[:, 1]
    d_top = h - points[:, 1]
    
    # Distance to other points
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    # dist_sq[i, i] = 0, so we need to handle that.
    # We want min_j (dist(i, j)) for j != i.
    # Set diagonal to inf
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq, axis=1)
    min_dist = np.sqrt(min_dist_sq)
    
    # Min distance to wall
    min_wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    
    # Radius is half of min(distance to neighbor, distance to wall)
    # Wait, if I place a circle at p, its radius is limited by:
    # 1. Distance to wall / 2? No, distance to wall is the max radius if alone.
    # 2. Distance to other circle / 2.
    # So r_i = 0.5 * min( dist_to_wall_i, min_j dist(i, j) )
    
    radii = 0.5 * np.minimum(min_wall_dist, min_dist)
    return radii

def _score(points, w, h):
    """Calculate sum of radii."""
    radii = _calc_radii(points, w, h)
    return np.sum(radii)

def _greedy_select(points, w, h, k=21):
    """Select k points greedily to maximize sum of radii."""
    selected = []
    candidates = list(range(len(points)))
    
    for _ in range(k):
        best_score = -1
        best_idx = -1
        for idx in candidates:
            # Test adding idx
            test_points = np.vstack([points[selected], points[idx:idx+1]])
            score = _score(test_points, w, h)
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx == -1: break
        selected.append(best_idx)
        candidates.remove(best_idx)
        
    return np.array(selected)

def _local_search(points, w, h, grid_res=30, max_iter=1000):
    """Local search to improve the set of points."""
    # Initialize with greedy selection
    # To make it faster, we can use a coarser grid for greedy, then refine?
    # Or just use the same grid.
    
    # Let's use a slightly larger grid for candidates to allow "moves"
    # But the candidate set for selection should be the same grid.
    
    selected_indices = _greedy_select(points, w, h, N)
    current_score = _score(points[selected_indices], w, h)
    
    # Precompute neighbors? Or just iterate.
    # Since grid is ~30x30=900 points, and N=21, 
    # Each iteration: for each of 21 points, try replacing with one of ~900 others?
    # 21 * 900 = 18900 evaluations per iteration.
    # Each evaluation is O(N^2) = 441 ops.
    # 18900 * 441 ~ 8 million ops.
    # 1000 iterations ~ 8 billion ops. Too slow in Python.
    
    # Optimization: Only try moving to nearby points in the grid.
    # Or, use a smarter local search.
    
    # Alternative: Use the continuous optimizer but initialized from this discrete solution?
    # The directive asks for a structural change to discrete optimization.
    # Let's optimize the local search.
    
    # We can vectorize the score calculation for a batch of candidates.
    # But the selection is sequential.
    
    # Let's limit the local search to a small number of iterations or a subset of candidates.
    # Or, use a different strategy: 
    # 1. Greedy select.
    # 2. For each selected point, find the best replacement among ALL candidates.
    # 3. Repeat until no improvement.
    
    # To speed up _score, we can precompute distances?
    # No, the set changes.
    
    # Let's try to make _score faster.
    # _calc_radii is O(N^2).
    # If we replace point i with point j:
    # We need to recalculate radii for all points.
    
    # Maybe we can use a simpler heuristic for local search:
    # Swap two points?
    
    # Let's stick to the greedy + simple local search with limited iterations.
    # If it's too slow, we'll reduce max_iter.
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            current_point_idx = selected_indices[i]
            best_score = current_score
            best_new_idx = -1
            
            # Try replacing with other candidates
            # To speed up, only try candidates that are "close" or random sample?
            # Let's try all candidates for now, but break early if possible? No.
            
            # Optimization: If current point is already at max possible radius for its location, 
            # moving might not help much.
            
            for j in range(len(points)):
                if j == current_point_idx: continue
                if j in selected_indices: continue # Can't pick same point twice
                
                # Swap i with j
                new_indices = selected_indices.copy()
                new_indices[i] = j
                new_score = _score(points[new_indices], w, h)
                
                if new_score > best_score:
                    best_score = new_score
                    best_new_idx = j
            
            if best_new_idx != -1:
                selected_indices[i] = best_new_idx
                current_score = best_score
                improved = True
        
        if not improved:
            break
            
    return points[selected_indices]

def _pack_discrete(w, h, start_time, deadline):
    """Run discrete packing for a given aspect ratio."""
    if time.time() > deadline:
        return None, 0
        
    # Generate candidates
    # Grid resolution: 20-40?
    # 20x20 = 400 points.
    # 30x30 = 900 points.
    # Let's use 25x25 = 625.
    points = _get_candidates(w, h, grid_res=25)
    
    # Local search
    best_points = _local_search(points, w, h, max_iter=50)
    
    if best_points is None or len(best_points) < N:
        return None, 0
        
    score = _score(best_points, w, h)
    
    # Construct circle array
    radii = _calc_radii(best_points, w, h)
    circles = np.column_stack([best_points, radii])
    
    return circles, score

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_global_score = 0
    best_global_result = None
    
    # Optimize aspect ratio W. H = 2 - W.
    # W in [0.5, 1.5] roughly.
    
    # Use Nelder-Mead or simple grid search + local refinement.
    # Nelder-Mead is good for 1D.
    
    def objective(w):
        h = 2.0 - w
        if h <= 0 or w <= 0:
            return -1000
        circles, score = _pack_discrete(w, h, start_time, deadline)
        if circles is None:
            return -1000
        return -score # Minimize negative score
    
    # Initial guess
    w0 = 1.0
    
    # Simple local search around w0
    best_w = w0
    best_score_val = 0
    
    # Grid search first to find a good region
    w_grid = np.linspace(0.6, 1.4, 20)
    for w in w_grid:
        if time.time() > deadline - 5:
            break
        h = 2.0 - w
        circles, score = _pack_discrete(w, h, start_time, deadline)
        if circles is not None and score > best_score_val:
            best_score_val = score
            best_w = w
            best_global_result = circles
    
    # Refine with local search on W
    # Use a few iterations of gradient-free optimization
    # Since _pack_discrete is noisy (local search might get stuck), 
    # we can just do a fine grid search around best_w.
    
    if time.time() < deadline - 5:
        w_refine = np.linspace(best_w - 0.1, best_w + 0.1, 20)
        for w in w_refine:
            if time.time() > deadline - 2:
                break
            h = 2.0 - w
            circles, score = _pack_discrete(w, h, start_time, deadline)
            if circles is not None and score > best_score_val:
                best_score_val = score
                best_w = w
                best_global_result = circles
    
    if best_global_result is None:
        # Fallback
        w = 1.0
        h = 1.0
        circles, score = _pack_discrete(w, h, start_time, deadline)
        if circles is not None:
            best_global_result = circles
    
    return best_global_result
