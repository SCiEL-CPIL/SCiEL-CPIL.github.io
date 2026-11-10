import time
import random
import itertools
from collections import defaultdict

# Precompute all C(30,3) triple indices once
_TRIPLE_INDICES = list(itertools.combinations(range(30), 3))

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area using precomputed indices."""
    counts = {}
    for i, j, k in _TRIPLE_INDICES:
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if d > 0:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def _score_with_counts(points):
    """Compute the maximum multiplicity and return (max_count, area_with_max)."""
    counts = {}
    for i, j, k in _TRIPLE_INDICES:
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if d > 0:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return 0, 0
    best_area = max(counts, key=counts.get)
    return counts[best_area], best_area

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

def _count_target(pts, target_area):
    """Count triples with exactly target_area."""
    cnt = 0
    for i, j, k in _TRIPLE_INDICES:
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        x3, y3 = pts[k]
        d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if d == target_area:
            cnt += 1
    return cnt

def _greedy_target_area(target_area, points_so_far, all_points_set, start_points, time_limit):
    """
    Greedily add points to maximize triangles with target_area.
    """
    start_time = time.time()
    pts = list(start_points)
    current_count = _count_target(pts, target_area)
    
    # Candidate pool
    pts_set = set(pts)
    candidates = [p for p in all_points_set if p not in pts_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try to swap a point in pts with a point in candidates
        idx = random.randint(0, len(pts) - 1)
        old_pt = pts[idx]
        
        # Try a few random candidates
        for _ in range(30):
            if not candidates:
                break
            new_pt = random.choice(candidates)
            if new_pt == old_pt:
                continue
            if new_pt in pts_set:
                continue
            
            # Swap
            pts[idx] = new_pt
            new_count = _count_target(pts, target_area)
            
            if new_count > current_count:
                current_count = new_count
                improved = True
                # Update sets
                pts_set.discard(old_pt)
                pts_set.add(new_pt)
                # Remove new_pt from candidates, add old_pt
                if new_pt in candidates:
                    candidates.remove(new_pt)
                candidates.append(old_pt)
            else:
                # Revert
                pts[idx] = old_pt
                
        if improved:
            # Occasionally rebuild candidate list
            if random.random() < 0.05:
                pts_set = set(pts)
                candidates = [p for p in all_points_set if p not in pts_set]
                random.shuffle(candidates)
                
    return pts, current_count

def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe (90s limit)
    
    # Initial candidates
    candidates = [
        _generate_grid(5, 6, -2, -2),
        _generate_grid(6, 5, -2, -2),
        _generate_grid(3, 10, -1, -4),
        _generate_grid(10, 3, -4, -1),
        _generate_grid(2, 15, 0, -7),
        _generate_grid(15, 2, -7, 0),
        _generate_grid(4, 8, -1, -3),
        _generate_grid(8, 4, -3, -1),
        _generate_grid(5, 5, -2, -2),
        _generate_grid(5, 5, -1, -1),
    ]
    
    valid_candidates = [c for c in candidates if _is_valid(c)]
    if not valid_candidates:
        return [(x, y) for x in range(6) for y in range(5)]
        
    # Evaluate initial candidates
    best_score = -1
    best_points = None
    best_area = None
    for c in valid_candidates:
        s, area = _score_with_counts(c)
        if s > best_score:
            best_score = s
            best_points = c[:]
            best_area = area
                
    current_points = best_points[:]
    current_score = best_score
    current_area = best_area
    
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    all_points_set = set(all_points)
    
    # Strategy 1: Targeted Area Optimization
    if best_area is not None and best_area > 0:
        # Give 15 seconds for this phase
        time_limit = 15.0
        opt_pts, opt_count = _greedy_target_area(best_area, current_points, all_points_set, current_points, time_limit)
        
        # Evaluate the optimized points with the full score
        new_score, new_area = _score_with_counts(opt_pts)
        if new_score > best_score:
            best_score = new_score
            best_points = opt_pts[:]
            current_points = opt_pts[:]
            current_score = new_score
            current_area = new_area
            
    # Strategy 2: Standard Local Search with precomputed indices
    # Use a set for O(1) membership checks
    pts_set = set(current_points)
    
    while time.time() - start_time < deadline:
        idx = random.randint(0, 29)
        old_pt = current_points[idx]
        
        new_pt = None
        # Rejection sampling
        for _ in range(15):
            candidate_pt = random.choice(all_points)
            if candidate_pt == old_pt:
                continue
            if candidate_pt in pts_set:
                continue
            new_pt = candidate_pt
            break
        
        if new_pt is None:
            continue
            
        current_points[idx] = new_pt
        pts_set.discard(old_pt)
        pts_set.add(new_pt)
        
        new_score, new_area = _score_with_counts(current_points)
        
        if new_score > current_score:
            current_score = new_score
            current_area = new_area
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        elif new_score == current_score:
            if random.random() < 0.5:
                pass
            else:
                current_points[idx] = old_pt
                pts_set.discard(new_pt)
                pts_set.add(old_pt)
        else:
            current_points[idx] = old_pt
            pts_set.discard(new_pt)
            pts_set.add(old_pt)
            
    return best_points
