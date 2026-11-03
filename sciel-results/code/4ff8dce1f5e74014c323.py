"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import cKDTree

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

def _generate_hex_candidates(r_target, margin=0.01):
    """
    Generate candidate centers on a hexagonal grid suitable for radius r_target.
    The spacing is chosen such that valid positions are at least 2*r_target apart.
    We generate a grid slightly denser than the theoretical limit to allow for
    boundary adjustments and local variations, but sparse enough to be manageable.
    Actually, for backtracking, we want a set of *potential* centers.
    A hex grid with spacing d = 2 * r_target * (1 - epsilon) is too dense for direct
    backtracking without heavy pruning.
    
    Instead, we use a standard hex grid with spacing d = 2 * r_target.
    Any two points on this grid are at least d apart (except nearest neighbors which are exactly d).
    Wait, nearest neighbors in hex grid are distance d. Next nearest are d*sqrt(3).
    So if we pick points from a hex grid with spacing 2*r, any subset is valid (distance >= 2r).
    The only constraint is the boundary.
    
    This simplifies the problem: We just need to select N points from a hex grid
    of spacing 2*r such that they fit in the square.
    If such a set exists, the radius is at least r.
    If we use spacing slightly less than 2*r, we might get violations.
    Let's stick to spacing = 2*r.
    
    To allow for "irregular" packings that might beat the regular grid, we could
    jitter the grid or use a slightly smaller spacing and prune.
    However, for a robust start, let's generate a hex grid with spacing 2*r.
    """
    d = 2 * r_target
    if d <= 0:
        return np.array([])
    
    # Hex grid parameters
    dx = d
    dy = d * np.sqrt(3) / 2
    
    # Determine grid size
    # x range: [-0.5, 0.5]
    # y range: [-0.5, 0.5]
    
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    
    candidates = []
    
    # Iterate over rows
    row_idx = 0
    y = y_min
    while y <= y_max:
        # Offset for even/odd rows
        offset = 0 if row_idx % 2 == 0 else dx / 2
        
        x = x_min + offset
        while x <= x_max:
            # Check boundary
            # The circle must be inside [-0.5, 0.5]^2
            # Center (x,y) is valid if 0.5 - max(abs(x), abs(y)) >= r_target
            # i.e., abs(x) <= 0.5 - r_target and abs(y) <= 0.5 - r_target
            if abs(x) <= 0.5 - r_target and abs(y) <= 0.5 - r_target:
                candidates.append((x, y))
            x += dx
        y += dy
        row_idx += 1
        
    if not candidates:
        return np.array([])
    return np.array(candidates)

def _backtrack_place(candidates, n_points, min_dist_sq, deadline):
    """
    Try to place n_points from candidates such that all pairwise distances >= min_dist_sq.
    Uses backtracking with a heuristic: pick the candidate with the most constraints (fewest available neighbors) first?
    Or simply greedy with backtracking on failure.
    
    For N=200 and ~200-300 candidates, this might be fast if the graph is sparse.
    """
    n_cand = len(candidates)
    if n_cand < n_points:
        return None
    
    # Precompute adjacency: which candidates are too close?
    # dist_sq < min_dist_sq implies conflict.
    # We want to pick a set of n_points with NO conflicts.
    
    # Build conflict graph (adjacency matrix of conflicts)
    # N_cand is small enough for a dense matrix if < 1000.
    # Let's use a list of sets for neighbors that are CONFLICTING.
    # Actually, we want to pick non-conflicting nodes.
    
    # Efficient check:
    # Sort candidates by some heuristic? 
    # Let's just do standard backtracking.
    
    selected = []
    used = [False] * n_cand
    
    # Heuristic: Order candidates by "degree" in the conflict graph?
    # High degree = more conflicts = harder to place.
    # Or low degree = easier to place.
    # Usually, "most constrained" first is better for CSP.
    # Let's compute the number of conflicts for each node.
    
    # To save time, we'll just use the natural order or random order.
    # Random order helps escape symmetries.
    
    order = list(range(n_cand))
    random.shuffle(order)
    
    # Precompute distances between all pairs? O(N_cand^2).
    # N_cand ~ 200-300. 300^2 = 90,000. Fast.
    dists_sq = np.zeros((n_cand, n_cand))
    for i in range(n_cand):
        diff = candidates[i] - candidates
        dists_sq[i] = np.sum(diff**2, axis=1)
        
    # Conflict matrix: True if dist_sq < min_dist_sq
    # We need dist >= min_dist, so conflict if dist_sq < min_dist_sq.
    # Note: min_dist_sq is (2*r)^2 = 4*r^2.
    # Wait, the score is r. The constraint is dist >= 2r.
    # So min_dist_sq should be (2*r)^2.
    
    conflicts = dists_sq < min_dist_sq
    np.fill_diagonal(conflicts, False) # Self is not a conflict
    
    # Backtracking
    # We need to select n_points nodes such that no two are connected in 'conflicts'.
    # This is finding an independent set of size n_points.
    
    # Simple recursive backtracking
    # To speed up, we can use bitsets if n_cand <= 64, but it's likely > 64.
    # Let's use a simple list of available indices.
    
    start_time = time.time()
    
    def backtrack(idx, current_selected):
        if time.time() > deadline:
            return None
            
        if len(current_selected) == n_points:
            return current_selected[:]
            
        # Pruning: if remaining candidates not enough
        remaining_slots = n_points - len(current_selected)
        # Count available candidates from idx onwards
        # This is O(N) per step, which is okay.
        
        for i in range(idx, n_cand):
            cand_idx = order[i]
            
            if used[cand_idx]:
                continue
                
            # Check if cand_idx conflicts with any already selected
            is_conflict = False
            for sel_idx in current_selected:
                if conflicts[cand_idx, sel_idx]:
                    is_conflict = True
                    break
                    
            if is_conflict:
                continue
            
            # Try placing cand_idx
            used[cand_idx] = True
            current_selected.append(cand_idx)
            
            result = backtrack(i + 1, current_selected)
            if result is not None:
                return result
                
            # Backtrack
            current_selected.pop()
            used[cand_idx] = False
            
        return None

    return backtrack(0, [])

def _local_polish(centers, deadline):
    """
    Simple local search to improve the radius of a valid packing.
    Since the backtracking gives a valid packing for a specific r,
    we can try to increase r slightly and re-optimize positions?
    Or just do a force-directed push-apart to increase the min distance.
    """
    centers = centers.copy()
    current_r = _score(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Use a simple gradient ascent on min distance
    # Move points away from their nearest neighbors.
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=2)
        min_dist = np.min(dists[:, 1])
        
        if min_dist / 2.0 > best_r:
            best_r = min_dist / 2.0
            best_centers = centers.copy()
            
        # Find the closest pair
        min_idx = np.argmin(dists[:, 1])
        i = min_idx
        j = idxs[i, 1]
        
        diff = centers[i] - centers[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-9:
            continue
        dir_vec = diff / norm
        
        # Move them apart
        step = 0.001
        centers[i] += dir_vec * step
        centers[j] -= dir_vec * step
        
        # Clamp
        centers = np.clip(centers, -0.5, 0.5)
        
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with a good estimate. Best known is ~0.0366.
    # Let's try to find a packing for r = 0.0365, 0.0366, etc.
    # We will binary search or linear search for the max r where a packing exists.
    
    # Initial guess
    r_low = 0.030
    r_high = 0.038
    
    best_centers = None
    best_score = 0.0
    
    # Try a few specific radii
    # To save time, we'll do a coarse search first.
    radii_to_try = [0.0366, 0.0365, 0.0364, 0.0360, 0.0350, 0.0340, 0.0330]
    
    for r in radii_to_try:
        if time.time() > deadline - 10:
            break
            
        candidates = _generate_hex_candidates(r)
        if len(candidates) < N:
            continue
            
        min_dist_sq = (2 * r) ** 2
        
        # Run backtracking
        result_indices = _backtrack_place(candidates, N, min_dist_sq, deadline - 10)
        
        if result_indices is not None:
            centers = candidates[result_indices]
            score = _score(centers)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                
            # If we found a valid packing, try to polish it
            if time.time() < deadline - 5:
                polished_centers, polished_score = _local_polish(centers, deadline - 5)
                if polished_score > best_score:
                    best_score = polished_score
                    best_centers = polished_centers.copy()
                    
            # If we found a valid packing for r, we can try a slightly higher r
            # But let's stick to the list for simplicity and robustness.
            # Breaking early if we hit the known best might be smart.
            if best_score > 0.0366:
                break

    # Fallback if backtracking failed
    if best_centers is None:
        # Generate a hex grid initialization as in the original
        r_est = 0.038
        dx = 2 * r_est
        dy = dx * np.sqrt(3) / 2
        
        best_points = None
        best_score = 0
        
        for n_rows in range(12, 18):
            n_cols = int(np.ceil(N / n_rows))
            if n_rows * n_cols < N:
                continue
                
            points = []
            for row in range(n_rows):
                y = row * dy
                offset = 0 if row % 2 == 0 else dx / 2
                for col in range(n_cols):
                    x = col * dx + offset
                    points.append((x, y))
            
            points = np.array(points[:N])
            
            center = points.mean(axis=0)
            points -= center
            
            max_extent = np.max(np.abs(points))
            if max_extent > 0:
                scale = 0.45 / max_extent
                points *= scale
                
            if np.all(np.abs(points) <= 0.5 - 1e-10):
                score = _score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
        if best_points is not None:
            best_centers = best_points
        else:
            # Last resort: square grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            best_centers = grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
