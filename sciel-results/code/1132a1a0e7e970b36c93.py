"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _generate_candidates(resolution=20):
    """
    Generate a dense set of candidate points on a hexagonal lattice.
    We create a grid of hex centers covering a slightly larger area than the square
    to ensure we have candidates near the optimal boundary positions.
    """
    # Estimate spacing. We want ~200 points in area 1.
    # Hex packing density ~0.9069.
    # Area per point ~ 1/200 = 0.005.
    # Hex cell area A = 2*sqrt(3)*r^2.
    # 2*sqrt(3)*r^2 ~ 0.005 => r ~ 0.038.
    # Spacing dx = 2r ~ 0.076. dy = sqrt(3)*r ~ 0.066.
    
    # To be safe and allow for "greedy" selection from a denser pool,
    # let's use a resolution factor.
    # Base spacing for 200 points in 1x1 square:
    # cols ~ sqrt(200/0.866) ~ 15.2. Let's say 16 cols.
    # spacing ~ 1/16 ~ 0.0625.
    
    # Let's generate a grid with spacing roughly half of the optimal spacing
    # to provide more freedom for the greedy selector to find better local arrangements
    # if the lattice is too coarse.
    
    # Actually, a simpler approach: Generate a fine square grid of candidates?
    # No, hex is better.
    
    # Let's generate a hex grid that covers [-0.6, 0.6]^2 to be safe,
    # with a spacing that allows for roughly 2x the number of points we need,
    # giving the greedy algorithm choices.
    
    target_spacing = 0.04  # Finer than optimal ~0.076, so we have ~4x more candidates
    dx = target_spacing
    dy = target_spacing * np.sqrt(3) / 2
    
    x_min, x_max = -0.6, 0.6
    y_min, y_max = -0.6, 0.6
    
    points = []
    row = 0
    y = y_min
    while y <= y_max:
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = x_min + x_offset
        while x <= x_max:
            points.append((x, y))
            x += dx
        y += dy
        row += 1
        
    candidates = np.array(points)
    # Filter candidates that are too close to the boundary such that
    # a circle of radius r_min would definitely be outside?
    # Actually, we just need them inside [-0.5, 0.5].
    # The score function handles the boundary constraint.
    # But candidates outside [-0.5, 0.5] will have negative boundary margin,
    # so they will never be selected if there are valid ones.
    # However, to save memory/computation, filter to [-0.5, 0.5].
    mask = (candidates[:, 0] >= -0.5) & (candidates[:, 0] <= 0.5) & \
           (candidates[:, 1] >= -0.5) & (candidates[:, 1] <= 0.5)
    candidates = candidates[mask]
    
    return candidates

def _greedy_select(candidates, n_select, deadline=None):
    """
    Greedily select n_select points from candidates to maximize the minimum
    distance to other selected points and the boundary.
    
    This is a standard farthest-point sampling variant adapted for max-min distance.
    At each step, we pick the candidate that maximizes the minimum distance
    to all *already selected* points and the boundary.
    
    Note: This is a heuristic. It tends to find good packings but might get stuck
    in suboptimal configurations if the initial choices are poor.
    To mitigate this, we can run it with different random seeds or initial points.
    """
    n_cand = len(candidates)
    if n_cand < n_select:
        return candidates[:n_select]
    
    selected = []
    selected_set = set()
    
    # Pre-compute boundary distances for all candidates
    # boundary_dist = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Actually, distance to boundary for a point (x,y) in [-0.5, 0.5] is:
    # min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y)
    # Which simplifies to 0.5 - max(|x|, |y|)
    boundary_dists = 0.5 - np.max(np.abs(candidates), axis=1)
    
    # We maintain an array `min_dist_so_far` for each candidate.
    # Initially, min_dist_so_far[i] = boundary_dists[i]
    # When we select a point p, we update min_dist_so_far[i] = min(min_dist_so_far[i], dist(i, p))
    
    min_dist_so_far = boundary_dists.copy()
    
    # To speed up selection, we can use a heap or just argmax.
    # N=200, Candidates ~ 1000-2000. O(N * N_cand^2) is 200 * 4e6 = 8e8 ops.
    # This might be slow in Python loops.
    # Vectorized update:
    # When we pick index `best_idx`, we compute distances from `candidates[best_idx]` to all `candidates`.
    # `dists = np.linalg.norm(candidates - candidates[best_idx], axis=1)`
    # `min_dist_so_far = np.minimum(min_dist_so_far, dists)`
    # This is O(N_cand) per selection. Total O(N * N_cand^2).
    # 200 * (2000^2) = 800 million flops. NumPy can handle this in seconds.
    
    for _ in range(n_select):
        if deadline and time.time() > deadline:
            break
            
        # Find the candidate with the maximum min_dist_so_far
        best_idx = np.argmax(min_dist_so_far)
        
        selected.append(best_idx)
        
        # Update min_dist_so_far for all other candidates
        # Distance from best_idx to all candidates
        diff = candidates - candidates[best_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        min_dist_so_far = np.minimum(min_dist_so_far, dists)
        
        # Set the selected point's distance to 0 or -inf so it's not picked again
        min_dist_so_far[best_idx] = -1.0
        
    selected_candidates = candidates[selected]
    return selected_candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # 1. Generate candidates
    # Use a resolution that provides enough candidates.
    # For 200 points, a spacing of 0.04 gives ~ (1/0.04)^2 / 0.866 ~ 720 candidates.
    # Let's try a few resolutions to be robust.
    
    best_centers = None
    best_score = 0.0
    
    # Try different candidate grid spacings
    # Finer grid = more options, but slower greedy step.
    # Spacing 0.04 is a good start.
    
    candidates = _generate_candidates(resolution=0.04)
    
    # Run greedy selection
    centers = _greedy_select(candidates, N, deadline=deadline - 5)
    
    # The greedy result might be good, but we can try to improve it with a local search
    # or just return it. Given the "discrete optimization" directive, let's stick to
    # the discrete selection primarily, but maybe run a second pass with a slightly
    # different grid or randomization if time permits.
    
    score = _score(centers)
    if score > best_score:
        best_score = score
        best_centers = centers
    
    # If we have time, try a randomized greedy (perturb the selection order slightly
    # or use a different grid offset)
    if time.time() < deadline - 20:
        # Shift the grid slightly to break symmetries
        offset_x = 0.01
        offset_y = 0.01
        
        # Regenerate candidates with offset
        # Re-use logic but with offset
        target_spacing = 0.04
        dx = target_spacing
        dy = target_spacing * np.sqrt(3) / 2
        
        x_min, x_max = -0.6 + offset_x, 0.6 + offset_x
        y_min, y_max = -0.6 + offset_y, 0.6 + offset_y
        
        points = []
        row = 0
        y = y_min
        while y <= y_max:
            x_offset = 0 if row % 2 == 0 else dx / 2
            x = x_min + x_offset
            while x <= x_max:
                points.append((x, y))
                x += dx
            y += dy
            row += 1
            
        candidates2 = np.array(points)
        mask = (candidates2[:, 0] >= -0.5) & (candidates2[:, 0] <= 0.5) & \
               (candidates2[:, 1] >= -0.5) & (candidates2[:, 1] <= 0.5)
        candidates2 = candidates2[mask]
        
        centers2 = _greedy_select(candidates2, N, deadline=deadline - 5)
        score2 = _score(centers2)
        
        if score2 > best_score:
            best_score = score2
            best_centers = centers2
            
    return best_centers
# EVOLVE-BLOCK-END
