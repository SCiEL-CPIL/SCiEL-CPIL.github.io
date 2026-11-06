"""Optimized local search with incremental area updates."""
import time
import numpy as np
import random

_DEADLINE = 75.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2


def _init_areas(pts):
    """
    Initialize the areas array for all triples.
    Returns:
        areas: np.array of shape (n, n, n) where areas[i,j,k] is |det| for i<j<k, else 0.
        triples: list of (i,j,k) tuples for i<j<k.
    """
    n = len(pts)
    p = np.asarray(pts, dtype=np.int64)
    areas = np.zeros((n, n, n), dtype=np.int64)
    triples = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = p[i]
                x2, y2 = p[j]
                x3, y3 = p[k]
                det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                a = abs(det)
                areas[i, j, k] = a
                areas[j, k, i] = a
                areas[k, i, j] = a
                triples.append((i, j, k))
                
    return areas, triples


def _get_max_multiplicity(areas, triples):
    """Calculate max multiplicity from areas array."""
    if not triples:
        return 0
    # Extract only valid areas
    vals = []
    for i, j, k in triples:
        a = areas[i, j, k]
        if a > 0:
            vals.append(a)
    if not vals:
        return 0
    # Use numpy for speed
    arr = np.array(vals)
    vals_unique, counts = np.unique(arr, return_counts=True)
    return int(counts.max())


def _local_refine_fast(pts, deadline):
    """
    Local search with incremental updates.
    """
    pts = [tuple(p) for p in pts]
    n = len(pts)
    
    # Precompute triples
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
                
    # Initialize areas
    areas, _ = _init_areas(pts)
    cur_best = _get_max_multiplicity(areas, triples)
    
    # Map point index to set of triples involving it? 
    # Or just iterate all triples and update if idx is involved.
    # Since n=30, triples ~ 4000. Updating 4000 entries is fast.
    
    improved = True
    no_improve_count = 0
    radius = 4
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        
        # Adjust radius if stuck
        if no_improve_count > 10:
            radius = 8
        elif no_improve_count > 20:
            radius = 15
        
        for idx in range(n):
            if time.time() >= deadline:
                break
            
            ox, oy = pts[idx]
            best_move = None
            best_val = cur_best
            best_new_areas = None
            best_new_pts = None
            
            # Generate candidates
            # To speed up, we can sample if radius is large, but let's try full first for small radius
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        # Check collision
                        if (nx, ny) in pts:
                            continue
                        candidates.append((nx, ny))
                        
            if not candidates:
                continue
                
            # If too many candidates, sample
            if len(candidates) > 100:
                candidates = random.sample(candidates, 100)
                
            for cand in candidates:
                nx, ny = cand
                
                # Calculate new areas for triples involving idx
                # We need to update areas[i,j,k] where idx is one of i,j,k
                # And recalculate max multiplicity
                
                # To avoid copying the whole array, we can track changes
                # But _get_max_multiplicity iterates all triples anyway.
                # Let's just update the array in place, then revert if not better.
                
                old_areas = []
                for i, j, k in triples:
                    if idx in (i, j, k):
                        old = areas[i, j, k]
                        old_areas.append((i, j, k, old))
                        
                        # Calculate new area
                        # Get the other two points
                        others = [x for x in (i, j, k) if x != idx]
                        p1 = pts[idx]
                        p2 = pts[others[0]]
                        p3 = pts[others[1]]
                        
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        # Wait, p1 is the OLD point. We need to use the NEW point (nx, ny)
                        x1, y1 = nx, ny
                        
                        det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                        new_a = abs(det)
                        
                        # Update symmetric entries
                        areas[i, j, k] = new_a
                        areas[j, k, i] = new_a
                        areas[k, i, j] = new_a
                        
                # Evaluate new state
                new_val = _get_max_multiplicity(areas, triples)
                
                if new_val > best_val:
                    best_val = new_val
                    best_move = cand
                    # Save the state to restore if we don't accept
                    # Actually, if we accept, we keep the changes.
                    # If we don't accept, we need to revert.
                    # Let's store the old areas to revert later if this candidate isn't the best.
                    # This is getting complex. 
                    # Alternative: Copy the areas array? 
                    # 30^3 = 27000 integers. 27000 * 8 bytes = 216 KB. 
                    # Copying 100 times per point is 21 MB. Acceptable?
                    # Yes, 100 copies * 216KB = 21MB. 
                    # But we do this for each point. 
                    # Let's just revert the changes for each candidate before trying the next.
                    
                    # Revert changes for this candidate
                    for i, j, k, old in old_areas:
                        areas[i, j, k] = old
                        areas[j, k, i] = old
                        areas[k, i, j] = old
                        
                    # We found a better move. Apply it permanently.
                    # We need to re-apply the changes? 
                    # No, we reverted. We should just apply the best move found.
                    
                else:
                    # Revert changes for this candidate
                    for i, j, k, old in old_areas:
                        areas[i, j, k] = old
                        areas[j, k, i] = old
                        areas[k, i, j] = old
            
            if best_move:
                # Apply the best move
                pts[idx] = best_move
                cur_best = best_val
                improved = True
                no_improve_count = 0
                # We need to update the areas array to reflect the new point.
                # We reverted all changes. So we must re-apply the changes for best_move.
                nx, ny = best_move
                for i, j, k in triples:
                    if idx in (i, j, k):
                        others = [x for x in (i, j, k) if x != idx]
                        p2 = pts[others[0]]
                        p3 = pts[others[1]]
                        
                        x1, y1 = nx, ny
                        x2, y2 = p2
                        x3, y3 = p3
                        
                        det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                        new_a = abs(det)
                        
                        areas[i, j, k] = new_a
                        areas[j, k, i] = new_a
                        areas[k, i, j] = new_a
                        
                break # Restart the loop over points? Or continue?
                # Restarting is safer to pick up interactions.
                # But we can also just continue to the next point.
                # Let's break to restart the outer loop for consistency.
    
    return pts, cur_best


def _grid_seed():
    """6x5 grid baseline."""
    return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]


def _two_line_seed():
    """Two parallel lines."""
    return [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]


def _hex_seed():
    """Hexagonal-like arrangement."""
    pts = []
    pts.append((0, 0))
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
        pts.append((dx*2, dy*2))
    for dx, dy in [(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2),(1,2),(-1,2),(1,-2),(-1,-2)]:
        pts.append((dx*2, dy*2))
    for x in range(-30, 31):
        for y in range(-30, 31):
            if len(pts) >= 30:
                break
            if (x, y) not in pts:
                pts.append((x, y))
        if len(pts) >= 30:
            break
    return pts[:30]


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    
    best_pts = None
    best_val = 0
    
    seeds = [
        _grid_seed(),
        _two_line_seed(),
        _hex_seed(),
    ]
    
    # Add some random seeds
    rng = random.Random(42)
    for _ in range(3):
        random_seed = []
        used = set()
        while len(random_seed) < 30:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y))
                random_seed.append((x, y))
        seeds.append(random_seed)
        
    for seed_pts in seeds:
        if time.time() >= deadline - 5:
            break
        if len(seed_pts) != 30:
            continue
            
        # Allocate time based on remaining budget
        # Give more time to earlier seeds if they are promising?
        # For now, distribute evenly.
        remaining_time = deadline - time.time()
        time_per_seed = remaining_time / max(1, len(seeds))
        seed_deadline = time.time() + time_per_seed
        
        pts, val = _local_refine_fast(seed_pts, seed_deadline)
        if val > best_val:
            best_val = val
            best_pts = pts
    
    # Final validation and padding
    if best_pts is None or best_val == 0:
        best_pts = _grid_seed()
        # Calculate val for fallback
        areas, triples = _init_areas(best_pts)
        best_val = _get_max_multiplicity(areas, triples)
        
    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    if len(out) < 30:
        for x in range(-30, 31):
            for y in range(-30, 31):
                if len(out) >= 30:
                    break
                if (x, y) not in seen:
                    seen.add((x, y))
                    out.append((x, y))
            if len(out) >= 30:
                break
    
    return out[:30]
