import time
import random
import numpy as np
from collections import Counter

def _calculate_score(points):
    n = len(points)
    area_counts = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                dx2 = x3 - x1
                dy2 = y3 - y1
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    area_counts[area] += 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def _area_counts_vectorized(points):
    """Compute all triangle doubled areas using numpy for speed."""
    pts = np.array(points, dtype=np.int32)
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            dx1 = pts[j, 0] - pts[i, 0]
            dy1 = pts[j, 1] - pts[i, 1]
            for k in range(j + 1, n):
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    areas.append(area)
    return areas

def _grid_candidates():
    cands = []
    for w, h in [(5,6),(6,5),(3,10),(10,3),(15,2),(2,15),(4,8),(8,4),(5,7),(7,5),(6,5),(5,5)]:
        pts = [(x,y) for x in range(w) for y in range(h)]
        if len(pts) > 30:
            pts = pts[:30]
        if len(pts) == 30 and len(set(pts)) == 30:
            cands.append(pts)
    # 5x5 + 5 extras
    pts = [(x,y) for x in range(5) for y in range(5)] + [(5,y) for y in range(5)]
    if len(set(pts))==30: cands.append(pts)
    return cands

def _count_target_fast(pts, target):
    """Count triangles with exactly target doubled area."""
    n = len(pts)
    cnt = 0
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i+1, n):
            x2, y2 = pts[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j+1, n):
                x3, y3 = pts[k]
                area = abs(dx1*(y3-y1) - dy1*(x3-x1))
                if area == target:
                    cnt += 1
    return cnt

def _targeted_greedy(target_area, time_limit, rng):
    """Greedily build 30 points maximizing triangles with doubled area = target_area."""
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    start = time.time()
    
    pts = []
    used = set()
    for _ in range(4):
        p = rng.choice(all_coords)
        while p in used:
            p = rng.choice(all_coords)
        pts.append(p)
        used.add(p)
    
    while len(pts) < 30 and time.time() - start < time_limit:
        best_gain = -1
        best_p = None
        n_pts = len(pts)
        pair_vecs = []
        for i in range(n_pts):
            x1, y1 = pts[i]
            for j in range(i+1, n_pts):
                x2, y2 = pts[j]
                pair_vecs.append((x2-x1, y2-y1, i, j))
        
        sample_size = min(300, 60*61)
        candidates = rng.sample(all_coords, sample_size)
        for p in candidates:
            if p in used:
                continue
            x3, y3 = p
            gain = 0
            for (dx1, dy1, i, j) in pair_vecs:
                x1, y1 = pts[i]
                area = abs(dx1 * (y3 - y1) - dy1 * (x3 - x1))
                if area == target_area:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        if best_p is None:
            for _ in range(100):
                p = rng.choice(all_coords)
                if p not in used:
                    best_p = p
                    break
            if best_p is None:
                break
        pts.append(best_p)
        used.add(best_p)
    
    best_pts = list(pts)
    best_cnt = _count_target_fast(pts, target_area)
    start2 = time.time()
    remaining = time_limit - (time.time() - start)
    while time.time() - start2 < remaining and remaining > 0:
        idx = rng.randint(0, 29)
        old = pts[idx]
        improved = False
        for _ in range(8):
            new = rng.choice(all_coords)
            if new in used and new != old:
                continue
            pts[idx] = new
            cnt = _count_target_fast(pts, target_area)
            if cnt > best_cnt:
                best_cnt = cnt
                best_pts = list(pts)
                if new != old:
                    used.discard(old)
                    used.add(new)
                improved = True
            else:
                pts[idx] = old
        if not improved and time.time() - start2 > remaining * 0.7:
            break
    return best_pts, best_cnt

def _get_best_area_and_count(points):
    """Return (best_area, best_count) for the point set."""
    n = len(points)
    area_counts = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                area = abs(dx1*(y3-y1) - dy1*(x3-x1))
                if area > 0:
                    area_counts[area] += 1
    if not area_counts:
        return 0, 0
    best_area = max(area_counts, key=area_counts.get)
    return best_area, area_counts[best_area]

def _fast_local_search(points, time_limit, rng):
    """
    Optimized local search.
    Instead of recalculating all O(N^3) areas for every candidate point,
    we maintain a Counter of area counts.
    When swapping a point, we only update the counts for triangles involving that point.
    Complexity per swap: O(N^2) instead of O(N^3).
    """
    start = time.time()
    n = 30
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    cur = list(points)
    used = set(cur)
    
    # Initialize area counts
    area_counts = Counter()
    for i in range(n):
        x1, y1 = cur[i]
        for j in range(i + 1, n):
            x2, y2 = cur[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = cur[k]
                area = abs(dx1 * (y3 - y1) - dy1 * (x3 - x1))
                if area > 0:
                    area_counts[area] += 1
    
    cur_max_count = max(area_counts.values()) if area_counts else 0
    best_pts = list(cur)
    best_score = cur_max_count
    best_area_counts = dict(area_counts)
    
    stuck = 0
    
    while time.time() - start < time_limit:
        idx = rng.randint(0, n - 1)
        old_p = cur[idx]
        improved = False
        
        # Try a few random candidates for this index
        for _ in range(10):
            new_p = rng.choice(all_coords)
            if new_p in used and new_p != old_p:
                continue
            
            # Calculate delta in counts
            # We need to remove contributions of old_p and add contributions of new_p
            
            # Helper to get area counts involving a specific point index
            def get_areas_involving(idx, pts):
                """Returns list of (area, count_delta) for triangles involving pts[idx].
                Actually, it's better to just iterate and update the counter directly.
                """
                pass # Not using this helper, doing it inline for speed
            
            # 1. Remove old_p's contributions
            # Triangles are (idx, j, k) where j,k != idx
            # We iterate over all pairs (j, k) from the other n-1 points
            # But careful: the triangle (i, j, k) is unique.
            # If we fix idx, the triangles are (idx, j, k) for j,k in [0..n-1], j!=idx, k!=idx, j<k?
            # No, the set of triangles involving point idx is { {idx, j, k} | j,k in [0..n-1], j!=idx, k!=idx, j<k }
            # Wait, if we just iterate all j,k != idx, we might double count or miss ordering?
            # The set of triangles containing point P is simply all pairs of other points.
            # Number of such triangles is C(n-1, 2).
            
            # Let's just iterate all j, k != idx.
            # For each pair (j, k), the triangle is (idx, j, k).
            # We calculate area and decrement count.
            
            # Optimization: Precompute vectors from old_p to all other points?
            # No, just iterate.
            
            for j in range(n):
                if j == idx: continue
                x2, y2 = cur[j]
                dx1 = x2 - old_p[0]
                dy1 = y2 - old_p[1]
                for k in range(j + 1, n):
                    if k == idx: continue
                    x3, y3 = cur[k]
                    dx2 = x3 - old_p[0]
                    dy2 = y3 - old_p[1]
                    area = abs(dx1 * dy2 - dy1 * dx2)
                    if area > 0:
                        area_counts[area] -= 1
                        if area_counts[area] == 0:
                            del area_counts[area]
            
            # 2. Add new_p's contributions
            # Temporarily set cur[idx] = new_p? Or just use new_p in calculation.
            # We need to calculate area of (idx, j, k) with new_p at idx.
            
            for j in range(n):
                if j == idx: continue
                x2, y2 = cur[j]
                dx1 = x2 - new_p[0]
                dy1 = y2 - new_p[1]
                for k in range(j + 1, n):
                    if k == idx: continue
                    x3, y3 = cur[k]
                    dx2 = x3 - new_p[0]
                    dy2 = y3 - new_p[1]
                    area = abs(dx1 * dy2 - dy1 * dx2)
                    if area > 0:
                        area_counts[area] += 1
                        
            new_max_count = max(area_counts.values()) if area_counts else 0
            
            if new_max_count > cur_max_count:
                # Accept move
                cur[idx] = new_p
                used.discard(old_p)
                used.add(new_p)
                cur_max_count = new_max_count
                
                if new_max_count > best_score:
                    best_score = new_max_count
                    best_pts = list(cur)
                    best_area_counts = dict(area_counts)
                
                improved = True
                stuck = 0
                break
            else:
                # Reject move: Revert counts
                # We have to reverse the changes.
                # It is safer to just re-calculate if we reject? No, that defeats the purpose.
                # We must carefully revert.
                
                # Revert: Remove new_p, Add old_p
                
                # Remove new_p contributions
                for j in range(n):
                    if j == idx: continue
                    x2, y2 = cur[j]
                    dx1 = x2 - new_p[0]
                    dy1 = y2 - new_p[1]
                    for k in range(j + 1, n):
                        if k == idx: continue
                        x3, y3 = cur[k]
                        dx2 = x3 - new_p[0]
                        dy2 = y3 - new_p[1]
                        area = abs(dx1 * dy2 - dy1 * dx2)
                        if area > 0:
                            area_counts[area] -= 1
                            if area_counts[area] == 0:
                                del area_counts[area]
                
                # Add old_p contributions
                for j in range(n):
                    if j == idx: continue
                    x2, y2 = cur[j]
                    dx1 = x2 - old_p[0]
                    dy1 = y2 - old_p[1]
                    for k in range(j + 1, n):
                        if k == idx: continue
                        x3, y3 = cur[k]
                        dx2 = x3 - old_p[0]
                        dy2 = y3 - old_p[1]
                        area = abs(dx1 * dy2 - dy1 * dx2)
                        if area > 0:
                            area_counts[area] += 1
                            
        if not improved:
            stuck += 1
            if stuck > 50:
                # Reset to best
                cur = list(best_pts)
                used = set(cur)
                # Rebuild counts (expensive but rare)
                area_counts = Counter()
                for i in range(n):
                    x1, y1 = cur[i]
                    for j in range(i + 1, n):
                        x2, y2 = cur[j]
                        dx1 = x2 - x1
                        dy1 = y2 - y1
                        for k in range(j + 1, n):
                            x3, y3 = cur[k]
                            area = abs(dx1 * (y3 - y1) - dy1 * (x3 - x1))
                            if area > 0:
                                area_counts[area] += 1
                cur_max_count = max(area_counts.values()) if area_counts else 0
                stuck = 0
                
    return best_pts, best_score

def _target_boost_local_search(points, target_area, time_limit, rng):
    """Local search that specifically tries to boost the count of target_area triangles."""
    start = time.time()
    best_pts = list(points)
    best_cnt = _count_target_fast(points, target_area)
    cur = list(points)
    cur_cnt = best_cnt
    all_coords = [(x,y) for x in range(-30,31) for y in range(-30,31)]
    used = set(cur)
    stuck = 0
    while time.time() - start < time_limit:
        idx = rng.randint(0, 29)
        old = cur[idx]
        improved = False
        for _ in range(6):
            new = rng.choice(all_coords)
            if new in used and new != old:
                continue
            cur[idx] = new
            cnt = _count_target_fast(cur, target_area)
            if cnt > cur_cnt:
                cur_cnt = cnt
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_pts = list(cur)
                    used.discard(old)
                    used.add(new)
                improved = True
                stuck = 0
                break
            else:
                cur[idx] = old
        if not improved:
            stuck += 1
            if stuck > 60:
                cur = list(best_pts)
                cur_cnt = best_cnt
                used = set(cur)
                stuck = 0
    return best_pts, best_cnt

def construct():
    start = time.time()
    deadline = start + 58.0
    rng = random.Random(42)
    
    # Phase 1: grid candidates
    best_pts = None
    best_score = -1
    for c in _grid_candidates():
        s = _calculate_score(c)
        if s > best_score:
            best_score = s
            best_pts = list(c)
    
    # Phase 2: targeted greedy for several target areas
    elapsed = time.time() - start
    remaining = deadline - elapsed - 8.0
    if remaining > 10.0:
        target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        per_target = remaining / len(target_areas)
        for ta in target_areas:
            if time.time() > deadline - 8.0:
                break
            tl = min(per_target, deadline - time.time() - 8.0)
            if tl < 1.5:
                break
            pts, cnt = _targeted_greedy(ta, tl, rng)
            s = _calculate_score(pts)
            if s > best_score:
                best_score = s
                best_pts = list(pts)
    
    # Phase 3: Fast local search (Optimized)
    remaining = deadline - time.time() - 2.0
    if remaining > 5.0:
        pts, s = _fast_local_search(best_pts, remaining, rng)
        if s > best_score:
            best_score = s
            best_pts = pts

    # Phase 4: Second pass of fast local search if time remains
    remaining = deadline - time.time() - 1.0
    if remaining > 3.0:
        pts, s = _fast_local_search(best_pts, remaining, rng)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    return best_pts
