import time
import numpy as np
import random
from collections import Counter

# Hard wall-clock budget for the whole construct() call (safety margin under 90s).
_DEADLINE = 85.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2
_N = 30


def _doubled_areas(pts):
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
    areas, _ = _doubled_areas(pts)
    return _best_multiplicity(areas)


def _init_counts(pts):
    """Initialize area counts histogram and return (counts, max_count)."""
    areas, _ = _doubled_areas(pts)
    counts = np.zeros(_MAX_AREA + 1, dtype=np.int64)
    pos_areas = areas[areas > 0]
    if pos_areas.size > 0:
        np.add.at(counts, pos_areas, 1)
    max_count = int(counts.max()) if counts.size > 0 else 0
    return counts, max_count


def _local_refine(pts, deadline, initial_radius=3, max_radius=10, random_trials=5):
    """
    Optimized local search with incremental area count updates.
    We maintain a histogram of areas to get O(1) score updates.
    """
    pts = [tuple(p) for p in pts]
    n = len(pts)
    
    # Precompute triples involving each point for fast updates
    # triples[i] is a list of (j, k) such that i < j < k or i > j > k etc.
    # Actually, we need to update all triples (i, j, k) where i is the moving point.
    # Let's precompute for each point i, the list of (j, k) pairs from the full set of triples.
    # Since we use i<j<k representation, if we move point i, we need to update:
    # 1. Triples where i is the first index: (i, j, k)
    # 2. Triples where i is the second index: (j, i, k)
    # 3. Triples where i is the third index: (j, k, i)
    
    # It is easier to just keep the full list of triples (i,j,k) and filter.
    # For n=30, total triples is 4495.
    # For a specific point, it is in C(29,2) = 406 triples.
    # We can precompute a list of (j, k) for each point i, and the type of role (1st, 2nd, 3rd)
    # to quickly compute the new area.
    
    point_triples = [[] for _ in range(n)]
    # Generate all triples i<j<k
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                point_triples[i].append((j, k, 'first'))
                point_triples[j].append((i, k, 'second'))
                point_triples[k].append((i, j, 'third'))
                
    # Initialize counts
    counts, cur_best = _init_counts(pts)
    
    pts_set = set(pts)
    improved = True
    no_improve_count = 0
    radius = initial_radius
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        
        # Adjust radius if stuck
        if no_improve_count > 3:
            radius = min(radius + 2, max_radius)
        elif no_improve_count > 8:
            radius = max_radius # Global-ish search
            
        for idx in range(n):
            if time.time() >= deadline:
                break
            
            ox, oy = pts[idx]
            best_move = None
            best_val = cur_best
            best_new_counts = None
            
            # Generate candidates
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) not in pts_set:
                            candidates.append((nx, ny))
                            
            # Add some random distant candidates if radius is large or we are stuck
            if radius >= max_radius or no_improve_count > 5:
                for _ in range(random_trials):
                    rx = random.randint(-30, 30)
                    ry = random.randint(-30, 30)
                    if (rx, ry) not in pts_set:
                        candidates.append((rx, ry))
                        
            if not candidates:
                continue
                
            # Sample if too many
            if len(candidates) > 150:
                candidates = random.sample(candidates, 150)
                
            for cand in candidates:
                nx, ny = cand
                new_counts = counts.copy()
                
                # Update counts for triples involving idx
                # We need to subtract old areas and add new areas.
                # We can do this in a loop over point_triples[idx]
                
                # To speed up, we can calculate the new areas for all triples involving idx
                # and then update the counts.
                
                # Optimization: We don't need to copy the whole counts array for every candidate
                # if we just want to check if the max increases.
                # But we need the exact max.
                # Since n is small, copying an array of size 4001 is fast.
                
                for j, k, role in point_triples[idx]:
                    # Get old area
                    if role == 'first':
                        i_idx, j_idx, k_idx = idx, j, k
                    elif role == 'second':
                        i_idx, j_idx, k_idx = j, idx, k
                    else:
                        i_idx, j_idx, k_idx = j, k, idx
                    
                    # We need the old area value.
                    # We didn't store the areas array in this function, only counts.
                    # We need to recompute the old area or store it.
                    # Let's store the areas in a dictionary or recompute.
                    # Recomputing is O(1).
                    
                    p1 = pts[i_idx]
                    p2 = pts[j_idx]
                    p3 = pts[k_idx]
                    
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    old_a = abs(det)
                    
                    if old_a > 0:
                        new_counts[old_a] -= 1
                        if new_counts[old_a] == 0:
                            # We can't delete from numpy array easily, but we can just let it be 0.
                            # np.unique on the whole array is slow.
                            # We are maintaining counts manually.
                            # We just need to track the max.
                            pass
                        
                    # Calculate new area
                    if role == 'first':
                        new_p1 = (nx, ny)
                        new_p2 = pts[j]
                        new_p3 = pts[k]
                    elif role == 'second':
                        new_p1 = pts[j]
                        new_p2 = (nx, ny)
                        new_p3 = pts[k]
                    else:
                        new_p1 = pts[j]
                        new_p2 = pts[k]
                        new_p3 = (nx, ny)
                    
                    x1, y1 = new_p1
                    x2, y2 = new_p2
                    x3, y3 = new_p3
                    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                    new_a = abs(det)
                    
                    if new_a > 0:
                        new_counts[new_a] += 1
                        
                # Calculate new max
                # Finding max in a numpy array is O(M) where M is _MAX_AREA.
                # This is fast.
                val = int(new_counts.max())
                
                if val > best_val:
                    best_val = val
                    best_move = cand
                    best_new_counts = new_counts
                    # We can break early if we find a very good move?
                    # No, let's find the best among candidates to avoid local minima.
                    
            if best_move:
                # Apply the best move
                pts[idx] = best_move
                pts_set.remove((ox, oy))
                pts_set.add(best_move)
                
                # Update global counts
                counts = best_new_counts
                cur_best = best_val
                improved = True
                no_improve_count = 0
                # We don't break out of the point loop immediately,
                # but we might want to restart the point loop to re-evaluate other points
                # with the new configuration.
                # However, just continuing is also fine.
                # Let's restart the point loop for a cleaner sweep.
                break
    
    return pts, cur_best


def _hexagonal_seed(n=30, cx=0, cy=0, scale=3):
    """Generate a hexagonal lattice seed of n points centered at (cx,cy)."""
    pts = []
    row_y = 0
    row_offset = 0
    while len(pts) < n:
        for x in range(-15, 16):
            px = cx + x * scale + (row_offset * scale // 2)
            py = cy + row_y * scale
            if -30 <= px <= 30 and -30 <= py <= 30 and (px, py) not in pts:
                pts.append((px, py))
            if len(pts) >= n:
                break
        row_y += 1
        row_offset = 1 - row_offset
    return pts[:n]


def _grid_seed(rows, cols, cx=0, cy=0, sx=3, sy=3):
    """Generate a rectangular grid seed centered at (cx,cy)."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - cols // 2) * sx
            y = cy + (r - rows // 2) * sy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts[:30]


def _two_line_seed(n_per_line=15, spacing=2):
    """Two parallel lines with evenly spaced points."""
    pts = []
    start_x = -n_per_line * spacing // 2
    for i in range(n_per_line):
        pts.append((start_x + i * spacing, 0))
        pts.append((start_x + i * spacing, 1))
    return pts[:30]


def _sheared_grid_seed(rows, cols, cx=0, cy=0, sx=3, sy=3, shear=1):
    """Generate a sheared grid seed: each row is shifted by shear * row_index."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = cx + (c - cols // 2) * sx + r * shear
            y = cy + (r - rows // 2) * sy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
    return pts[:30]


def _clustered_random_seed(n=30, cx=0, cy=0, spread=5, seed=42):
    """Generate a clustered random seed: points concentrated near center."""
    rng = random.Random(seed)
    pts = set()
    while len(pts) < n:
        x = rng.randint(cx - spread, cx + spread)
        y = rng.randint(cy - spread, cy + spread)
        if -30 <= x <= 30 and -30 <= y <= 30:
            pts.add((x, y))
    return list(pts)[:n]


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    random.seed(42)

    # Multiple diverse seeds
    seeds = [
        ("grid_6x5", [(x, y) for x in range(-2, 4) for y in range(-2, 3)]),
        ("grid_5x6", _grid_seed(5, 6, cx=0, cy=0, sx=3, sy=3)),
        ("hex_scale3", _hexagonal_seed(30, 0, 0, scale=3)),
        ("hex_scale4", _hexagonal_seed(30, 0, 0, scale=4)),
        ("two_lines", _two_line_seed(15, spacing=2)),
        ("grid_10x3", _grid_seed(10, 3, cx=0, cy=0, sx=5, sy=3)),
        ("sheared_6x5", _sheared_grid_seed(6, 5, cx=0, cy=0, sx=3, sy=3, shear=1)),
        ("sheared_5x6", _sheared_grid_seed(5, 6, cx=0, cy=0, sx=3, sy=3, shear=-1)),
        ("clustered", _clustered_random_seed(30, 0, 0, spread=5, seed=42)),
        ("clustered2", _clustered_random_seed(30, 0, 0, spread=4, seed=123)),
    ]

    best_pts = None
    best_val = 0

    # Phase 1: Quick hill-climb from each seed with moderate radius
    per_seed_budget = (deadline - start) / len(seeds)
    for name, seed in seeds:
        if time.time() >= deadline - 2:
            break
        seed_pts = [tuple(p) for p in seed[:30]]
        # Ensure exactly 30 distinct points in range
        seen = set()
        valid_seed = []
        for p in seed_pts:
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in seen:
                seen.add(p)
                valid_seed.append(p)
        # Pad if needed
        if len(valid_seed) < 30:
            for x in range(-30, 31):
                for y in range(-30, 31):
                    if len(valid_seed) >= 30:
                        break
                    if (x, y) not in seen:
                        seen.add((x, y))
                        valid_seed.append((x, y))
        valid_seed = valid_seed[:30]

        seed_deadline = time.time() + per_seed_budget
        pts, val = _local_refine(valid_seed, seed_deadline, initial_radius=3, max_radius=8)
        
        if val > best_val:
            best_val = val
            best_pts = pts

    # Phase 2: If we have time, do a second pass on the best found so far with wider search
    if best_pts and time.time() < deadline - 5:
        remaining_time = deadline - time.time()
        # Allocate half the remaining time to this final polish
        polish_deadline = time.time() + remaining_time * 0.5
        pts, val = _local_refine(best_pts, polish_deadline, initial_radius=5, max_radius=15, random_trials=10)
        if val > best_val:
            best_val = val
            best_pts = pts

    # Final validation and padding
    if best_pts is None or best_val == 0:
        best_pts = _grid_seed(6, 5, cx=0, cy=0, sx=3, sy=3)
        best_val = _evaluate(best_pts)
        
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
