"""Greedy set-cover approach targeting specific area values, with local refinement."""
import time
import numpy as np
import random
from itertools import combinations

_DEADLINE = 75.0
_MAX_AREA = 4000  # Safe upper bound for |det| in [-30,30]^2


def _doubled_areas_full(pts):
    """Return areas = |det| for each triple (i,j,k) with i<j<k."""
    n = len(pts)
    p = np.asarray(pts, dtype=np.int64)
    ix, jx, kx = np.indices((n, n, n))
    mask = (ix < jx) & (jx < kx)
    i = ix[mask].astype(np.int64)
    j = jx[mask].astype(np.int64)
    k = kx[mask].astype(np.int64)
    xi, yi = p[i, 0], p[i, 1]
    xj, yj = p[j, 0], p[j, 1]
    xk, yk = p[k, 0], p[k, 1]
    det = (xi * (yj - yk)) + (xj * (yk - yi)) + (xk * (yi - yj))
    areas = np.abs(det)
    return areas


def _best_multiplicity(areas):
    """Largest multiplicity of a single positive doubled area."""
    pos = areas[areas > 0]
    if pos.size == 0:
        return 0
    vals, counts = np.unique(pos, return_counts=True)
    return int(counts.max())


def _evaluate(pts):
    areas = _doubled_areas_full(pts)
    return _best_multiplicity(areas)


def _area_of_triple(p1, p2, p3):
    """Compute absolute determinant for three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    return abs(det)


def _greedy_target_area(target_area, deadline, seed=0):
    """
    Greedy algorithm to find 30 points maximizing triples with area = target_area.
    Strategy: iteratively add points that create the most new triples with target area.
    """
    rng = random.Random(seed)
    
    # Start with a few random points
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    rng.shuffle(all_points)
    
    # Initialize with 3 random points
    selected = all_points[:3]
    selected_set = set(selected)
    
    # Count how many triples with target area exist among selected points
    # We'll track this incrementally
    
    # For each candidate point, compute how many new target-area triples it would create
    # with pairs of already-selected points
    
    while len(selected) < 30 and time.time() < deadline:
        best_gain = -1
        best_point = None
        
        # Sample a subset of candidates for speed
        candidates = [p for p in all_points if p not in selected_set]
        if not candidates:
            break
        
        # For speed, sample at most 200 candidates
        if len(candidates) > 200:
            candidates = rng.sample(candidates, 200)
        
        for cand in candidates:
            # Count new target-area triples: for each pair (i,j) in selected,
            # check if area(selected[i], selected[j], cand) == target_area
            gain = 0
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if _area_of_triple(selected[i], selected[j], cand) == target_area:
                        gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_point = cand
        
        if best_point is None or best_gain <= 0:
            # No point improves, try a random one
            remaining = [p for p in all_points if p not in selected_set]
            if remaining:
                best_point = rng.choice(remaining)
            else:
                break
        
        selected.append(best_point)
        selected_set.add(best_point)
        
        # Early termination check
        if time.time() >= deadline - 5:
            break
    
    # Pad if needed
    if len(selected) < 30:
        remaining = [p for p in all_points if p not in selected_set]
        rng.shuffle(remaining)
        for p in remaining:
            if len(selected) >= 30:
                break
            selected.append(p)
            selected_set.add(p)
    
    return selected[:30]


def _local_refine(pts, deadline):
    """Local search to improve area multiplicity."""
    pts = [tuple(p) for p in pts]
    n = len(pts)
    cur_best = _evaluate(pts)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() >= deadline:
                break
            ox, oy = pts[idx]
            
            # Try nearby moves
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if -30 <= nx <= 30 and -30 <= ny <= 30:
                        if (nx, ny) in pts:
                            continue
                        
                        # Test the move
                        new_pts = pts.copy()
                        new_pts[idx] = (nx, ny)
                        val = _evaluate(new_pts)
                        
                        if val > cur_best:
                            cur_best = val
                            pts = new_pts
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return pts, cur_best


def _multi_start_greedy(deadline):
    """Try multiple target areas and initial seeds."""
    best_pts = None
    best_val = 0
    
    # Common target areas to try
    target_areas = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 36, 40, 48, 60]
    
    # Also try areas based on grid structure
    for a in range(1, 50):
        if a not in target_areas:
            target_areas.append(a)
    
    rng = random.Random(42)
    rng.shuffle(target_areas)
    
    # Limit number of attempts based on time
    max_attempts = 20
    
    for i, ta in enumerate(target_areas[:max_attempts]):
        if time.time() >= deadline - 10:
            break
        
        # Try with different seeds
        for seed in [0, 1, 2]:
            if time.time() >= deadline - 5:
                break
            
            pts = _greedy_target_area(ta, deadline - 5, seed=seed)
            if len(pts) != 30:
                continue
            
            val = _evaluate(pts)
            if val > best_val:
                best_val = val
                best_pts = pts
    
    return best_pts, best_val


def _grid_seed():
    """6x5 grid baseline."""
    return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]


def _two_line_seed():
    """Two parallel lines."""
    return [(x, 0) for x in range(-7, 8)] + [(x, 1) for x in range(-7, 8)]


def _hex_seed():
    """Hexagonal-like arrangement."""
    pts = []
    # Center
    pts.append((0, 0))
    # Ring 1
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
        pts.append((dx*2, dy*2))
    # Ring 2
    for dx, dy in [(2,0),(-2,0),(0,2),(0,-2),(2,2),(2,-2),(-2,2),(-2,-2),(1,2),(-1,2),(1,-2),(-1,-2)]:
        pts.append((dx*2, dy*2))
    # Pad to 30
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
    
    # Phase 1: Try grid-based seeds with local refinement
    seeds = [
        _grid_seed(),
        _two_line_seed(),
        _hex_seed(),
    ]
    
    for seed_pts in seeds:
        if time.time() >= deadline - 15:
            break
        if len(seed_pts) != 30:
            continue
        pts, val = _local_refine(seed_pts, deadline - 10)
        if val > best_val:
            best_val = val
            best_pts = pts
    
    # Phase 2: Multi-start greedy targeting specific areas
    greedy_pts, greedy_val = _multi_start_greedy(deadline - 10)
    if greedy_val > best_val:
        best_val = greedy_val
        best_pts = greedy_pts
    
    # Phase 3: Final local refinement on best result
    if best_pts is not None and time.time() < deadline - 2:
        refined_pts, refined_val = _local_refine(best_pts, deadline - 1)
        if refined_val > best_val:
            best_val = refined_val
            best_pts = refined_pts
    
    # Fallback: if we haven't found anything good, use grid
    if best_pts is None or best_val == 0:
        best_pts = _grid_seed()
        best_val = _evaluate(best_pts)
    
    # Final validation and padding
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
