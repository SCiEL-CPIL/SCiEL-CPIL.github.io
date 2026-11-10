import time
import random
import itertools
from collections import Counter

_START = time.time()
_DEADLINE = 78.0  # safety margin before 90s kill

def _det(p1, p2, p3):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _score(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d = _det(points[i], points[j], points[k])
                if d > 0:
                    areas.append(d)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _greedy_build(target_area, all_points, n_points=30):
    """Greedy: pick points that maximize triangles with target_area among selected."""
    selected = []
    remaining = list(all_points)
    # Start with a random seed triple that has target area
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
            # Count how many new target-area triangles this point forms with selected
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

def _local_search(points, all_points, max_iter=200):
    """Swap-based local search to improve the score."""
    current = list(points)
    current_score = _score(current)
    selected_set = set(current)
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(len(current)):
            if time.time() > _DEADLINE:
                break
            for j in range(len(all_points)):
                if all_points[j] in selected_set:
                    continue
                # Try swapping current[i] with all_points[j]
                new_points = current[:i] + [all_points[j]] + current[i+1:]
                new_score = _score(new_points)
                if new_score > current_score:
                    current = new_points
                    current_score = new_score
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
    
    # Pre-compute area histogram for small target areas
    # For efficiency, sample a subset of triples to find promising target areas
    random.seed(42)
    
    # Sample many random triples to find common small areas
    area_counts = Counter()
    sample_size = 50000
    for _ in range(sample_size):
        i = random.randint(0, len(all_points)-1)
        j = random.randint(0, len(all_points)-1)
        k = random.randint(0, len(all_points)-1)
        if i == j or j == k or i == k:
            continue
        d = _det(all_points[i], all_points[j], all_points[k])
        if d > 0 and d <= 200:
            area_counts[d] += 1
    
    # Get top candidate target areas
    if area_counts:
        top_areas = [a for a, c in area_counts.most_common(15) if c > 0]
    else:
        top_areas = list(range(2, 50, 2))  # even areas are common for integer points
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Greedy build for each top target area
    for target_area in top_areas[:8]:
        if time.time() > deadline - 5:
            break
        pts = _greedy_build(target_area, all_points, 30)
        if len(pts) == 30:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
    
    # Strategy 2: Local search from best greedy result
    if best_points and time.time() < deadline - 5:
        pts, s = _local_search(best_points, all_points, max_iter=100)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Strategy 3: Random restarts with local search
    restarts = 0
    while time.time() < deadline - 10 and restarts < 20:
        restarts += 1
        random_pts = _random_points(30)
        # Quick local search
        pts, s = _local_search(random_pts, all_points, max_iter=50)
        if s > best_score:
            best_score = s
            best_points = pts
    
    # Strategy 4: Structured constructions
    # Grid-based: points on a regular grid often give many equal-area triangles
    for gx in range(5, 9):
        if time.time() > deadline - 3:
            break
        gy = max(1, 30 // gx)
        grid_pts = []
        for x in range(-30, 31, max(1, 60//gx)):
            for y in range(-30, 31, max(1, 60//gy)):
                grid_pts.append((x, y))
        if len(grid_pts) >= 30:
            random.shuffle(grid_pts)
            pts = grid_pts[:30]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
            # Local search from grid
            if time.time() < deadline - 3:
                pts2, s2 = _local_search(pts, all_points, max_iter=30)
                if s2 > best_score:
                    best_score = s2
                    best_points = pts2
    
    # Strategy 5: Try specific known good constructions
    # Points forming a lattice pattern
    constructions = [
        # 6x5 grid centered
        [(x, y) for x in range(-3, 3) for y in range(-2, 3)],
        # 5x6 grid
        [(x, y) for x in range(-2, 3) for y in range(-3, 3)],
        # 3x10
        [(x, y) for x in range(-1, 2) for y in range(-4, 6)],
        # 10x3
        [(x, y) for x in range(-4, 6) for y in range(-1, 2)],
    ]
    
    for cons in constructions:
        if time.time() > deadline - 2:
            break
        if len(cons) >= 30:
            pts = cons[:30]
            # Check they're in range
            if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
    
    # Final local search on best
    if best_points and time.time() < deadline - 2:
        pts, s = _local_search(best_points, all_points, max_iter=50)
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
