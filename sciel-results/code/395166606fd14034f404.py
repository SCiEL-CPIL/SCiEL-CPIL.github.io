"""Combine grid baseline with vectorized hill-climbing refinement using incremental updates."""
import time
import numpy as np
import random

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 75.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2


def _doubled_areas_full(pts):
    """Return (areas, tri_idx): areas = |det| for each triple, tri_idx = (i,j,k) arrays.

    Vectorized over all C(n,3) triples. n is small (30) so this is cheap.
    """
    n = len(pts)
    p = np.asarray(pts, dtype=np.int64)
    ix, jx, kx = np.indices((n, n, n))
    # keep strictly i<j<k
    mask = (ix < jx) & (jx < kx)
    i = ix[mask].astype(np.int64)
    j = jx[mask].astype(np.int64)
    k = kx[mask].astype(np.int64)
    xi, yi = p[i, 0], p[i, 1]
    xj, yj = p[j, 0], p[j, 1]
    xk, yk = p[k, 0], p[k, 1]
    det = (xi * (yj - yk)) + (xj * (yk - yi)) + (xk * (yi - yj))
    areas = np.abs(det)
    return areas, (i, j, k)


def _best_multiplicity(areas):
    """Largest multiplicity of a single positive doubled area."""
    pos = areas[areas > 0]
    if pos.size == 0:
        return 0
    vals, counts = np.unique(pos, return_counts=True)
    return int(counts.max())


def _evaluate(pts):
    areas, _ = _doubled_areas_full(pts)
    return _best_multiplicity(areas)


def _init_counts(pts):
    """Initialize area counts histogram and return (counts, max_count)."""
    areas, _ = _doubled_areas_full(pts)
    counts = np.zeros(_MAX_AREA + 1, dtype=np.int64)
    # Filter out 0 areas for the "positive area" metric, but we store all to be safe?
    # The task says "exclude zero-area triples". So we only count area > 0.
    pos_areas = areas[areas > 0]
    if pos_areas.size > 0:
        np.add.at(counts, pos_areas, 1)
    max_count = int(counts.max()) if counts.size > 0 else 0
    return counts, max_count


def _hill_climb(pts, deadline):
    """Local search with incremental area count updates."""
    pts = [tuple(p) for p in pts]
    n = len(pts)
    cur_set = set(pts)
    
    # Precompute all pairs (i, j) with i < j. 
    # When moving point k, we need to update areas for triples (i, j, k) where i,j != k.
    # It's easier to just iterate i in range(n), j in range(n) where i != k and j != k and i < j.
    
    counts, cur_best = _init_counts(pts)
    cur_best_val = cur_best

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each point
        for idx in range(n):
            if time.time() >= deadline:
                break
            ox, oy = pts[idx]
            
            # Candidate moves: nearby cells
            cands = []
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in cur_set:
                            cands.append((nx, ny))
            
            # If no nearby moves, try a few random ones
            if not cands:
                for _ in range(10):
                    c = (random.randint(-30, 30), random.randint(-30, 30))
                    if c not in cur_set:
                        cands.append(c)
            
            for c in cands:
                if time.time() >= deadline:
                    break
                
                nx, ny = c
                # Check if move is valid (distinct)
                if (nx, ny) in cur_set:
                    continue
                
                # Calculate changes in area counts
                # Triples involving idx: (i, j, idx) with i < j and i,j != idx
                # We need to iterate over all pairs (i, j) from the other n-1 points.
                
                # Optimization: Precompute list of other indices?
                # Or just loop i, j in range(n) and skip idx.
                
                changes = [] # list of (old_area, new_area)
                for i in range(n):
                    if i == idx: continue
                    xi, yi = pts[i]
                    for j in range(i + 1, n):
                        if j == idx: continue
                        # Ensure i < j for unique triple identification relative to idx?
                        # The triple is {i, j, idx}.
                        # We just need the area value.
                        xj, yj = pts[j]
                        
                        # Old area
                        det_old = (xi * (yj - oy)) + (xj * (oy - yi)) + (ox * (yi - yj))
                        a_old = abs(det_old)
                        
                        # New area
                        det_new = (xi * (yj - ny)) + (xj * (ny - yi)) + (nx * (yi - yj))
                        a_new = abs(det_new)
                        
                        if a_old != a_new:
                            changes.append((a_old, a_new))
                
                if not changes:
                    continue
                
                # Apply changes to counts
                # We need to check if this improves the max count.
                # Instead of updating the global counts and checking max (O(MaxArea)),
                # we can track the counts of the specific areas involved.
                
                # However, the max count might be for an area NOT involved in the change.
                # So we need to know if the new max of the changed areas > cur_best.
                # And if the old max was for an area that decreased, we need to know the new max.
                
                # Efficient check:
                # 1. Identify unique areas in changes.
                # 2. For each such area, calculate new count.
                # 3. Find max of these new counts.
                # 4. If max_new_count > cur_best, it's an improvement.
                # 5. If max_new_count == cur_best, we might still accept if it allows future improvements?
                #    Standard hill climbing accepts only if >.
                
                # But wait, if cur_best was held by an area that DECREASED, and no other area increased above cur_best,
                # then the new max might be lower than cur_best.
                # We need to calculate the true new max.
                
                # Let's just update a temporary copy of the relevant counts?
                # No, the number of distinct areas is small.
                # Let's collect the areas involved.
                involved_areas = set()
                for a_old, a_new in changes:
                    if a_old > 0: involved_areas.add(a_old)
                    if a_new > 0: involved_areas.add(a_new)
                
                # Calculate current counts for these areas
                # And new counts
                max_candidate = 0
                for a in involved_areas:
                    old_cnt = counts[a]
                    delta = 0
                    for a_old, a_new in changes:
                        if a_old == a: delta -= 1
                        if a_new == a: delta += 1
                    new_cnt = old_cnt + delta
                    if new_cnt > max_candidate:
                        max_candidate = new_cnt
                
                # The new global max is max(max_candidate, cur_best_of_unchanged_areas)
                # We don't have cur_best_of_unchanged_areas easily.
                # But if max_candidate > cur_best, we definitely improve.
                # If max_candidate <= cur_best, it's possible the true max dropped if the area holding cur_best was decreased.
                # To be safe and simple, we only accept if max_candidate > cur_best.
                # This might miss some moves that maintain cur_best but change configuration,
                # but it ensures we never decrease the score.
                
                if max_candidate > cur_best:
                    # Apply changes
                    for a_old, a_new in changes:
                        if a_old > 0:
                            counts[a_old] -= 1
                        if a_new > 0:
                            counts[a_new] += 1
                    cur_best = max_candidate
                    pts[idx] = (nx, ny)
                    cur_set.remove((ox, oy))
                    cur_set.add((nx, ny))
                    improved = True
                    break # Restart loop for next point? Or continue?
                    # Breaking out of candidate loop for this point is good.
                    # But we should probably restart the outer loop to try other points again?
                    # Or just continue to next point.
        
    return pts, cur_best


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    # Baseline: 6x5 grid
    base = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    best_pts = [tuple(p) for p in base]
    best_val = _evaluate(best_pts)

    # Hill-climb from the baseline
    pts, val = _hill_climb(best_pts, deadline)
    if val > best_val:
        best_pts, best_val = pts, val

    # Try alternative seeds
    if time.time() < deadline - 5:
        # Seed 2: Two parallel lines
        seed2 = [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]
        p2, v2 = _hill_climb(seed2, deadline)
        if v2 > best_val:
            best_pts, best_val = p2, v2

    if time.time() < deadline - 5:
        # Seed 3: Points on a circle (approx)
        # Use integer points close to circle radius 15
        seed3 = []
        for x in range(-15, 16):
            y_sq = 225 - x*x
            if y_sq >= 0:
                y1 = int(np.sqrt(y_sq))
                if y1*y1 == y_sq:
                    seed3.append((x, y1))
                    if y1 > 0:
                        seed3.append((x, -y1))
        # Ensure 30 distinct points
        seen = set()
        unique_seed3 = []
        for p in seed3:
            if p not in seen:
                seen.add(p)
                unique_seed3.append(p)
        if len(unique_seed3) >= 30:
            unique_seed3 = unique_seed3[:30]
        else:
            # Pad with random
            while len(unique_seed3) < 30:
                p = (random.randint(-30, 30), random.randint(-30, 30))
                if p not in seen:
                    seen.add(p)
                    unique_seed3.append(p)
        
        p3, v3 = _hill_climb(unique_seed3, deadline)
        if v3 > best_val:
            best_pts, best_val = p3, v3

    # Final guard
    seen = set()
    out = []
    for p in best_pts:
        x, y = p
        if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in seen:
            seen.add((x, y))
            out.append((x, y))
    if len(out) < 30:
        for c in [(x, y) for x in range(-30, 31) for y in range(-30, 31)]:
            if len(out) >= 30:
                break
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out[:30]
