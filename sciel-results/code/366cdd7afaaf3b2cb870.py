"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

# Precompute all unique triple indices for n=30
_TRIPLES = []
for i in range(30):
    for j in range(i+1, 30):
        for k in range(j+1, 30):
            _TRIPLES.append((i, j, k))

def _compute_areas_and_counts(points):
    """
    Compute all doubled triangle areas and return (max_count, counts_dict, areas_list).
    Uses vectorized numpy for speed.
    """
    n = len(points)
    if n < 3:
        return 0, {}, []
    pts = np.array(points, dtype=np.int64)
    x1, y1 = pts[:, 0], pts[:, 1]
    
    # Extract coordinates for all triples
    i_idx = np.array([t[0] for t in _TRIPLES])
    j_idx = np.array([t[1] for t in _TRIPLES])
    k_idx = np.array([t[2] for t in _TRIPLES])
    
    x1_f = x1[i_idx]
    y1_f = y1[i_idx]
    x2_f = x1[j_idx]
    y2_f = y1[j_idx]
    x3_f = x1[k_idx]
    y3_f = y1[k_idx]
    
    dets = np.abs(x1_f * (y2_f - y3_f) - x2_f * (y1_f - y3_f) + x3_f * (y1_f - y2_f))
    
    # Filter out zero areas
    valid_mask = dets > 0
    valid_dets = dets[valid_mask]
    
    if len(valid_dets) == 0:
        return 0, {}, []
        
    # Count occurrences
    # Using np.bincount for speed if range is small, but areas can be large.
    # Counter on list is usually fine for 4495 items.
    counts = Counter(valid_dets.tolist())
    max_count = max(counts.values())
    
    return max_count, counts, valid_dets.tolist()

def _update_counts_incremental(points, idx, old_pos, new_pos, counts, areas_list):
    """
    Update the counts dict and areas_list when point idx moves from old_pos to new_pos.
    Returns new max_count.
    """
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    
    # Identify triples involving idx
    # We need to recalculate areas for all triples containing idx
    # And remove the old areas, add new areas
    
    # To keep it simple and fast, we will just recalculate the specific triples
    # and update the Counter.
    
    # First, find indices of triples in _TRIPLES that involve idx
    # And get their old areas from areas_list? 
    # It's easier to just recompute the affected areas and update the counter.
    
    # Let's identify which triples are affected.
    # A triple (i,j,k) is affected if i==idx or j==idx or k==idx.
    
    # We can precompute the list of triple indices that involve each point.
    # But for now, let's just iterate through _TRIPLES.
    
    # Actually, a faster way:
    # 1. Calculate old areas for affected triples.
    # 2. Decrease counts for those areas.
    # 3. Calculate new areas for affected triples.
    # 4. Increase counts for those areas.
    
    # Get coordinates of other points
    # We need to be careful with the indices in the triple.
    
    affected_old_areas = []
    affected_new_areas = []
    
    # Precompute coordinates for fast access
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # Temporarily set new position for calculation
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    # We will iterate through _TRIPLES
    for t_idx, (i, j, k) in enumerate(_TRIPLES):
        if i == idx or j == idx or k == idx:
            # Calculate old area
            # Replace idx with old_pos
            xi, yi = (old_x, old_y) if i == idx else (xs[i], ys[i])
            xj, yj = (old_x, old_y) if j == idx else (xs[j], ys[j])
            xk, yk = (old_x, old_y) if k == idx else (xs[k], ys[k])
            
            area_old = abs(xi*(yj-yk) - xj*(yi-yk) + xk*(yi-yj))
            if area_old > 0:
                affected_old_areas.append(area_old)
            
            # Calculate new area
            # Replace idx with new_pos
            xi, yi = (new_x, new_y) if i == idx else (xs[i], ys[i])
            xj, yj = (new_x, new_y) if j == idx else (xs[j], ys[j])
            xk, yk = (new_x, new_y) if k == idx else (xs[k], ys[k])
            
            area_new = abs(xi*(yj-yk) - xj*(yi-yk) + xk*(yi-yj))
            if area_new > 0:
                affected_new_areas.append(area_new)
            # If area_new is 0, it just disappears from counts (which is handled by decrementing old and not incrementing new)

    # Update counts
    for a in affected_old_areas:
        counts[a] -= 1
        if counts[a] == 0:
            del counts[a]
            
    for a in affected_new_areas:
        if a in counts:
            counts[a] += 1
        else:
            counts[a] = 1
            
    # Update areas_list? 
    # areas_list is not strictly needed for the score, only counts.
    # But if we want to reuse it, we should update it.
    # For simplicity, we don't maintain areas_list in the loop, 
    # we only use it for initialization or final verification.
    # The score is derived from counts.
    
    if not counts:
        return 0
        
    return max(counts.values())

def _local_search_best_improvement(points, time_limit=10):
    """
    Local search that tries to move each point to any valid position in the grid
    that improves the max multiplicity.
    """
    start_time = time.time()
    n = len(points)
    
    # Initialize counts
    _, counts, _ = _compute_areas_and_counts(points)
    current_max = max(counts.values()) if counts else 0
    
    # Precompute valid positions for each point?
    # Or just iterate over grid.
    grid_range = range(-30, 31)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        for idx in range(n):
            if time.time() - start_time > time_limit:
                break
                
            old_pos = points[idx]
            best_new_pos = None
            best_new_score = current_max
            
            # Try all possible new positions
            # To speed up, we can skip positions that are too far or similar, 
            # but for n=30, 3721 positions * 30 points * ~500 triples is ~55M ops.
            # Python is slow. We need to be careful.
            
            # Optimization: Only try positions that are not already occupied.
            occupied = set(points)
            
            # Heuristic: Try a subset of positions?
            # Or use numpy to vectorize the score calculation for all candidates at once?
            # That might be too complex.
            
            # Let's try a random sampling of positions if full search is too slow.
            # But "Best Improvement" requires checking all.
            
            # Alternative: Check if moving to a position that creates a specific area is better.
            # This is hard to predict.
            
            # Let's stick to a limited search: 
            # 1. Try all positions in a small neighborhood first?
            # 2. Or just sample 100 random positions?
            
            # Given the time limit, let's try a hybrid:
            # If n is small, full search. But 3721 * 30 is 111k evaluations.
            # Each evaluation takes ~1ms? 111s. Too slow.
            
            # We must reduce the candidate set.
            # Let's sample 50 random positions per point per iteration.
            
            candidates = set()
            # Add current position's neighbors
            ox, oy = old_pos
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        candidates.add((nx, ny))
            
            # Add random positions
            for _ in range(50):
                candidates.add((random.randint(-30, 30), random.randint(-30, 30)))
            
            for new_pos in candidates:
                if new_pos == old_pos:
                    continue
                if new_pos in occupied:
                    continue
                
                # Calculate new score
                # We need to simulate the move
                # We can use _update_counts_incremental
                # But we need to restore counts if we don't accept.
                # So we copy counts? Counter copy is O(K). K is small.
                # Or we can just calculate the new max directly from the affected areas?
                
                # Let's use the incremental update but on a copy of the counts.
                # This is safer.
                
                # Actually, _update_counts_incremental modifies the dict in place.
                # So we must copy it.
                
                # To avoid deep copy overhead, we can just compute the new areas for the affected triples
                # and determine the new max count for those specific areas, 
                # assuming other areas don't change their max.
                # Wait, if we decrease the count of the current max area, the new max might be lower.
                # If we increase the count of another area, it might become the new max.
                
                # Let's just do the full incremental update on a copied Counter.
                # Counter copy is fast.
                
                trial_counts = counts.copy()
                new_max = _update_counts_incremental(points, idx, old_pos, new_pos, trial_counts, None)
                
                if new_max > best_new_score:
                    best_new_score = new_max
                    best_new_pos = new_pos
            
            if best_new_pos is not None:
                # Accept the move
                points[idx] = best_new_pos
                # Update the actual counts
                _update_counts_incremental(points, idx, old_pos, best_new_pos, counts, None)
                current_max = best_new_score
                improved = True
                
    return points, current_max

def construct():
    deadline = time.time() + 55  # Safety margin before 90s limit
    
    # Generate a few initial configurations
    initial_configs = []
    
    # Grid
    pts = []
    for x in range(6):
        for y in range(5):
            pts.append((x, y))
    initial_configs.append(pts)
    
    # Random
    for _ in range(5):
        pts = set()
        while len(pts) < 30:
            pts.add((random.randint(-30, 30), random.randint(-30, 30)))
        initial_configs.append(list(pts))
        
    best_score = 0
    best_points = None
    
    for init_points in initial_configs:
        if time.time() > deadline:
            break
        
        score, _, _ = _compute_areas_and_counts(init_points)
        if score > best_score:
            best_score = score
            best_points = list(init_points)
            
    if best_points is None:
        best_points = initial_configs[0]
        best_score, _, _ = _compute_areas_and_counts(best_points)
        
    # Local search
    # We have ~50s left.
    # Let's run local search for a fixed time, or multiple times if fast.
    
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        # Run local search
        # We can run it multiple times with different random seeds if it's fast
        # But _local_search_best_improvement is deterministic except for random candidates.
        
        # Let's run it once with a generous time limit.
        improved_points, improved_score = _local_search_best_improvement(best_points, time_limit=remaining_time - 5)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_points
            
    return best_points
# EVOLVE-BLOCK-END
