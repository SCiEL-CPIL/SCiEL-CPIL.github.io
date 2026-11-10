import time
import random
import itertools
import numpy as np
from collections import Counter

_START = time.time()
_DEADLINE = 78.0  # safety margin before 90s kill

def _det(p1, p2, p3):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _score_np(pts_array):
    """Vectorized score: largest multiplicity of a single positive doubled area.
    pts_array: numpy array of shape (n, 2) with integer coords."""
    n = pts_array.shape[0]
    if n < 3:
        return 0
    # Compute all pairwise differences
    # For triangle (i,j,k): det = |xi*(yj-yk) + xj*(yk-yi) + xk*(yi-yj)|
    # Vectorized: for each pair (i,j), compute det with all k > j
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorized over k
            dx = pts_array[j, 0] - pts_array[i, 0]
            dy = pts_array[j, 1] - pts_array[i, 1]
            # det for (i, j, k) = |dx * (yk - yi) - dy * (xk - xi)|
            dets = np.abs(dx * (pts_array[j+1:, 1] - pts_array[i, 1]) - 
                          dy * (pts_array[j+1:, 0] - pts_array[i, 0]))
            # Filter positive
            pos = dets[dets > 0]
            if len(pos) > 0:
                areas.append(pos)
    if not areas:
        return 0
    all_areas = np.concatenate(areas)
    if len(all_areas) == 0:
        return 0
    # Count multiplicities
    vals, counts = np.unique(all_areas, return_counts=True)
    return int(counts.max())

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    arr = np.array(points, dtype=np.int32)
    return _score_np(arr)

def _greedy_build(target_area, all_points, n_points=30):
    """Greedy: pick points that maximize triangles with target_area among selected."""
    selected = []
    remaining = list(all_points)
    random.shuffle(remaining)
    found_seed = False
    for i in range(len(remaining)):
        for j in range(i+1, len(remaining)):
            for k in range(j+1, len(remaining)):
                if _det(remaining[i], remaining[j], remaining[k]) == target_area:
                    selected = [remaining[i], remaining[j], remaining[k]]
                    found_seed = True
                    break
            if found_seed:
                break
        if found_seed:
            break
    if not found_seed:
        selected = remaining[:3]
    
    remaining = [p for p in remaining if p not in selected]
    
    while len(selected) < n_points and time.time() < _DEADLINE:
        best_pt = None
        best_gain = -1
        for pt in remaining:
            gain = 0
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if _det(pt, selected[i], selected[j]) == target_area:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_pt = pt
        if best_pt is None:
            break
        selected.append(best_pt)
        remaining.remove(best_pt)
    
    return selected

def _local_search_fast(points, all_points_np, all_points_list, max_iter=500):
    """Fast swap-based local search using vectorized scoring."""
    current = list(points)
    current_arr = np.array(current, dtype=np.int32)
    current_score = _score_np(current_arr)
    selected_set = set(current)
    n = len(current)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        # Try each position
        for i in range(n):
            if time.time() > _DEADLINE:
                break
            # Try swapping with a random subset of candidates for speed
            candidates = [p for p in all_points_list if p not in selected_set]
            if not candidates:
                break
            random.shuffle(candidates)
            # Limit candidates per position for speed
            max_cand = min(len(candidates), 200)
            for j in range(max_cand):
                new_pt = candidates[j]
                # Build new array
                new_arr = current_arr.copy()
                new_arr[i] = new_pt
                new_score = _score_np(new_arr)
                if new_score > current_score:
                    current[i] = new_pt
                    current_arr[i] = new_pt
                    current_score = new_score
                    selected_set.discard(current[i-1] if i > 0 else None)
                    # Rebuild set
                    selected_set = set(current)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current, current_score

def _random_points(n=30):
    pts = set()
    while len(pts) < n:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        pts.add((x, y))
    return list(pts)

def construct():
    global _START
    start = time.time()
    deadline = start + 78.0
    
    # All candidate points in [-30,30]^2
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    all_points_np = np.array(all_points, dtype=np.int32)
    all_points_set = set(all_points)
    
    random.seed(42)
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Structured grid constructions (fast, good starting points)
    grid_constructions = []
    # 6x5 grid
    grid_constructions.append([(x, y) for x in range(-3, 3) for y in range(-2, 3)])
    # 5x6 grid
    grid_constructions.append([(x, y) for x in range(-2, 3) for y in range(-3, 3)])
    # 10x3
    grid_constructions.append([(x, y) for x in range(-4, 6) for y in range(-1, 2)])
    # 3x10
    grid_constructions.append([(x, y) for x in range(-1, 2) for y in range(-4, 6)])
    # 7x5 with some adjustment
    grid_constructions.append([(x, y) for x in range(-3, 4) for y in range(-2, 3)][:30])
    # 8x4
    grid_constructions.append([(x, y) for x in range(-3, 5) for y in range(-2, 2)][:30])
    # 15x2
    grid_constructions.append([(x, y) for x in range(-7, 8) for y in range(0, 2)][:30])
    # 20x2 (even)
    grid_constructions.append([(x, y) for x in range(-9, 11) for y in range(0, 2)][:30])
    # 12x3
    grid_constructions.append([(x, y) for x in range(-5, 7) for y in range(-1, 2)][:30])
    # 13x3
    grid_constructions.append([(x, y) for x in range(-6, 7) for y in range(-1, 2)][:30])
    # 25x2
    grid_constructions.append([(x, y) for x in range(-12, 13) for y in range(0, 2)][:30])
    # 30x2
    grid_constructions.append([(x, y) for x in range(-14, 16) for y in range(0, 2)][:30])
    # 11x3
    grid_constructions.append([(x, y) for x in range(-5, 6) for y in range(-1, 2)][:30])
    # 9x4
    grid_constructions.append([(x, y) for x in range(-4, 5) for y in range(-2, 2)][:30])
    # 14x3
    grid_constructions.append([(x, y) for x in range(-6, 8) for y in range(-1, 2)][:30])
    # 16x2
    grid_constructions.append([(x, y) for x in range(-7, 9) for y in range(0, 2)][:30])
    # 18x2
    grid_constructions.append([(x, y) for x in range(-8, 10) for y in range(0, 2)][:30])
    # 20x2 odd
    grid_constructions.append([(x, y) for x in range(-9, 11) for y in range(-1, 1)][:30])
    # 22x2
    grid_constructions.append([(x, y) for x in range(-10, 12) for y in range(0, 2)][:30])
    # 24x2
    grid_constructions.append([(x, y) for x in range(-11, 13) for y in range(0, 2)][:30])
    # 26x2
    grid_constructions.append([(x, y) for x in range(-12, 14) for y in range(0, 2)][:30])
    # 28x2
    grid_constructions.append([(x, y) for x in range(-13, 15) for y in range(0, 2)][:30])
    # 30x2
    grid_constructions.append([(x, y) for x in range(-14, 16) for y in range(0, 2)][:30])
    
    for cons in grid_constructions:
        if time.time() > deadline - 5:
            break
        if len(cons) >= 30:
            pts = cons[:30]
            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                pts = list(set(pts))
                if len(pts) == 30:
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
    
    # Strategy 2: Greedy build for target areas
    # Sample to find common areas
    area_counts = Counter()
    for _ in range(30000):
        i = random.randint(0, len(all_points)-1)
        j = random.randint(0, len(all_points)-1)
        k = random.randint(0, len(all_points)-1)
        if i == j or j == k or i == k:
            continue
        d = _det(all_points[i], all_points[j], all_points[k])
        if d > 0 and d <= 100:
            area_counts[d] += 1
    
    if area_counts:
        top_areas = [a for a, c in area_counts.most_common(10) if c > 0]
    else:
        top_areas = list(range(2, 30, 2))
    
    for target_area in top_areas[:6]:
        if time.time() > deadline - 10:
            break
        pts = _greedy_build(target_area, all_points, 30)
        if len(pts) == 30:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
    
    # Strategy 3: Local search from best
    if best_points and time.time() < deadline - 5:
        pts, s = _local_search_fast(best_points, all_points_np, all_points, max_iter=300)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Strategy 4: Multiple random restarts with local search
    restarts = 0
    while time.time() < deadline - 8 and restarts < 30:
        restarts += 1
        random_pts = _random_points(30)
        pts, s = _local_search_fast(random_pts, all_points_np, all_points, max_iter=100)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Strategy 5: Final polish on best
    if best_points and time.time() < deadline - 2:
        pts, s = _local_search_fast(best_points, all_points_np, all_points, max_iter=200)
        if s > best_score:
            best_score = s
            best_points = pts
    
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    # Ensure exactly 30 distinct points in range
    best_points = list(set(best_points))[:30]
    while len(best_points) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) not in best_points:
            best_points.append((x, y))
    
    return best_points
