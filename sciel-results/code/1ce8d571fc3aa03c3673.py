import time
import random
from itertools import combinations

_DEADLINE = 80.0  # safety margin under 90s
_START = None

def _doubled_area(p, q, r):
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def _score(points, target_area):
    """Count triples with doubled area == target_area."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _doubled_area(points[i], points[j], points[k]) == target_area:
                    count += 1
    return count

def _score_all(points):
    """Return (best_count, best_area) over all positive areas."""
    n = len(points)
    from collections import Counter
    areas = Counter()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas[a] += 1
    if not areas:
        return 0, 0
    best_area, best_count = areas.most_common(1)[0]
    return best_count, best_area

def _random_points(n=30, lo=-30, hi=30):
    pts = set()
    while len(pts) < n:
        pts.add((random.randint(lo, hi), random.randint(lo, hi)))
    return list(pts)

def _hill_climb(points, target_area, time_limit):
    """Local search: move one point at a time to maximize count of triples with area==target_area."""
    global _START
    best = _score(points, target_area)
    current = list(points)
    n = len(current)
    deadline = _START + time_limit
    stuck = 0
    while time.time() < deadline:
        improved = False
        # try moving a random point to a random location
        idx = random.randint(0, n-1)
        new_pt = (random.randint(-30, 30), random.randint(-30, 30))
        if new_pt in current and new_pt != current[idx]:
            continue
        old_pt = current[idx]
        current[idx] = new_pt
        # check no duplicates
        if len(set(current)) < n:
            current[idx] = old_pt
            continue
        new_score = _score(current, target_area)
        if new_score > best:
            best = new_score
            improved = True
            stuck = 0
        else:
            current[idx] = old_pt
            stuck += 1
        if stuck > 200:
            break
    return current, best

def _try_target_area(points, target_area, time_limit):
    """Hill-climb for a specific target area."""
    global _START
    deadline = _START + time_limit
    best_pts = list(points)
    best_count = _score(points, target_area)
    
    while time.time() < deadline:
        # pick a random point and try small perturbations
        idx = random.randint(0, len(points)-1)
        ox, oy = points[idx]
        found = False
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                if (nx, ny) in points and (nx, ny) != points[idx]:
                    continue
                old = points[idx]
                points[idx] = (nx, ny)
                if len(set(points)) < len(points):
                    points[idx] = old
                    continue
                sc = _score(points, target_area)
                if sc > best_count:
                    best_count = sc
                    best_pts = list(points)
                    found = True
                else:
                    points[idx] = old
            if found:
                break
        if not found and random.random() < 0.1:
            # occasional larger jump
            points[idx] = (random.randint(-30,30), random.randint(-30,30))
            if len(set(points)) < len(points):
                points[idx] = best_pts[idx]
    
    return best_pts, best_count

def construct():
    global _START
    _START = time.time()
    
    best_overall_count = 0
    best_overall_pts = None
    
    # Strategy: try several target areas, hill-climb for each
    # Small areas like 1, 2, 3 tend to have more triangles
    target_areas = [1, 2, 3, 4, 5, 6]
    
    time_per_target = 8.0  # seconds per target area
    
    for ta in target_areas:
        if time.time() - _START > _DEADLINE - 5:
            break
        time_limit = min(time_per_target, _DEADLINE - (time.time() - _START) - 2)
        if time_limit < 1:
            break
        
        # Start from a few different initial configurations
        for init_iter in range(3):
            if time.time() - _START > _DEADLINE - 3:
                break
            
            # Different initializations
            if init_iter == 0:
                # Grid-like start
                pts = [(x, y) for x in range(-2, 4) for y in range(-2, 3)][:30]
            elif init_iter == 1:
                # Hexagonal-ish
                pts = []
                for row in range(-3, 4):
                    for col in range(-3, 4):
                        x = col + (row % 2) * 0  # simple grid for now
                        y = row
                        pts.append((x, y))
                pts = pts[:30]
            else:
                pts = _random_points(30)
            
            # Ensure distinct
            pts = list(dict.fromkeys(pts))[:30]
            while len(pts) < 30:
                pts.append((random.randint(-30,30), random.randint(-30,30)))
                pts = list(dict.fromkeys(pts))[:30]
            
            pts, count = _try_target_area(pts, ta, time_limit / 3.0)
            if count > best_overall_count:
                best_overall_count = count
                best_overall_pts = list(pts)
    
    # Final polish: if we have a good candidate, do more refinement
    if best_overall_pts and best_overall_count > 0:
        remaining = _DEADLINE - (time.time() - _START) - 2
        if remaining > 2:
            # Find the actual best area for this configuration
            bc, ba = _score_all(best_overall_pts)
            if bc > best_overall_count:
                best_overall_count = bc
            # Refine for the actual best area
            best_overall_pts, refined_count = _try_target_area(best_overall_pts, ba, remaining)
            if refined_count > best_overall_count:
                best_overall_count = refined_count
    
    # Validate and return
    if best_overall_pts is None:
        return [(x, y) for x in range(6) for y in range(5)]
    
    # Ensure exactly 30 distinct points in range
    result = list(dict.fromkeys(best_overall_pts))
    result = [p for p in result if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
    while len(result) < 30:
        result.append((random.randint(-30,30), random.randint(-30,30)))
        result = list(dict.fromkeys(result))[:30]
    
    return result[:30]
