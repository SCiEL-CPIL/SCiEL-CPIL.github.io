import time
import random
import itertools
from collections import defaultdict, Counter

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area."""
    counts = Counter()
    n = len(points)
    for i in range(n - 2):
        x1, y1 = points[i]
        for j in range(i + 1, n - 1):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if d > 0:
                    counts[d] += 1
    if not counts:
        return 0
    return max(counts.values())

def _generate_grid(w, h, offset_x=0, offset_y=0):
    pts = []
    for x in range(w):
        for y in range(h):
            pts.append((x + offset_x, y + offset_y))
    return pts

def _is_valid(pts):
    if len(pts) != 30:
        return False
    if len(set(pts)) != 30:
        return False
    return all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts)

def construct():
    start_time = time.time()
    deadline = 55.0  # 55 seconds to be safe
    
    # Initial candidates: various grids
    candidates = [
        _generate_grid(5, 6, -2, -2),
        _generate_grid(6, 5, -2, -2),
        _generate_grid(3, 10, -1, -4),
        _generate_grid(10, 3, -4, -1),
        _generate_grid(2, 15, 0, -7),
        _generate_grid(15, 2, -7, 0),
        _generate_grid(4, 8, -1, -3),
        _generate_grid(8, 4, -3, -1),
    ]
    
    valid_candidates = [c for c in candidates if _is_valid(c)]
    if not valid_candidates:
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Evaluate initial candidates
    best_score = -1
    best_points = None
    best_area = None
    for c in valid_candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c[:]
            # Find the area that achieved this score
            counts = Counter()
            n = len(c)
            for i in range(n - 2):
                x1, y1 = c[i]
                for j in range(i + 1, n - 1):
                    x2, y2 = c[j]
                    for k in range(j + 1, n):
                        x3, y3 = c[k]
                        d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if d > 0:
                            counts[d] += 1
            if counts:
                best_area = max(counts, key=counts.get)
                
    current_points = best_points[:]
    current_score = best_score
    
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    all_points_set = set(all_points)
    
    # Strategy 1: Targeted Area Optimization using greedy approach
    if best_area is not None and time.time() - start_time < deadline - 10:
        target_area = best_area
        time_limit = 10.0
        
        def _count_target(pts):
            cnt = 0
            n = len(pts)
            for i in range(n - 2):
                x1, y1 = pts[i]
                for j in range(i + 1, n - 1):
                    x2, y2 = pts[j]
                    for k in range(j + 1, n):
                        x3, y3 = pts[k]
                        d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if d == target_area:
                            cnt += 1
            return cnt
        
        current_count = _count_target(current_points)
        candidates_pool = [p for p in all_points_set if p not in current_points]
        random.shuffle(candidates_pool)
        
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            idx = random.randint(0, len(current_points) - 1)
            old_pt = current_points[idx]
            
            for _ in range(30):
                if not candidates_pool:
                    break
                new_pt = random.choice(candidates_pool)
                if new_pt == old_pt:
                    continue
                
                current_points[idx] = new_pt
                new_count = _count_target(current_points)
                
                if new_count > current_count:
                    current_count = new_count
                    improved = True
                    # Update pool
                    if old_pt in candidates_pool:
                        candidates_pool.remove(old_pt)
                    candidates_pool.append(new_pt)
                    random.shuffle(candidates_pool)
                    break
                else:
                    current_points[idx] = old_pt
        
        new_score = _score(current_points)
        if new_score > best_score:
            best_score = new_score
            best_points = current_points[:]
            current_points = best_points[:]
            current_score = best_score
    
    # Strategy 2: Standard Local Search with smarter moves
    while time.time() - start_time < deadline:
        idx = random.randint(0, 29)
        old_pt = current_points[idx]
        
        new_pt = None
        # Rejection sampling with bias toward nearby points
        for _ in range(15):
            # Try a local move first
            if random.random() < 0.5:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                candidate_pt = (old_pt[0] + dx, old_pt[1] + dy)
                if -30 <= candidate_pt[0] <= 30 and -30 <= candidate_pt[1] <= 30:
                    if candidate_pt not in current_points:
                        new_pt = candidate_pt
                        break
            else:
                candidate_pt = random.choice(all_points)
                if candidate_pt not in current_points:
                    new_pt = candidate_pt
                    break
        
        if new_pt is None:
            continue
            
        current_points[idx] = new_pt
        new_score = _score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        elif new_score == current_score:
            if random.random() < 0.3:
                pass
            else:
                current_points[idx] = old_pt
        else:
            current_points[idx] = old_pt
            
    return best_points
