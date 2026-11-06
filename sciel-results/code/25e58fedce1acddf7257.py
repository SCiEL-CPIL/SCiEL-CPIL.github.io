import time
import random
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

def _targeted_greedy(target_area, time_limit, rng):
    """Greedily build 30 points maximizing triangles with doubled area = target_area."""
    # Precompute: for a target area A, which pairs of points produce it?
    # We build incrementally. For each candidate new point, count how many new
    # triangles with area A it forms with existing points.
    all_coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    start = time.time()
    
    # Start with a few random points, then greedily add
    pts = []
    used = set()
    # Seed with 3-5 random points
    for _ in range(4):
        p = rng.choice(all_coords)
        while p in used:
            p = rng.choice(all_coords)
        pts.append(p)
        used.add(p)
    
    while len(pts) < 30 and time.time() - start < time_limit:
        # For each candidate, count new triangles with area = target_area
        best_gain = -1
        best_p = None
        # Sample candidates for speed
        n_pts = len(pts)
        # Precompute pair vectors for existing points
        pair_vecs = []
        for i in range(n_pts):
            x1, y1 = pts[i]
            for j in range(i+1, n_pts):
                x2, y2 = pts[j]
                pair_vecs.append((x2-x1, y2-y1, i, j))
        
        # Sample a subset of candidates
        sample_size = min(200, 60*61)
        candidates = rng.sample(all_coords, sample_size)
        for p in candidates:
            if p in used:
                continue
            x3, y3 = p
            gain = 0
            for (dx1, dy1, i, j) in pair_vecs:
                # triangle (pts[i], pts[j], p)
                # area = |dx1 * (y3-y1) - dy1 * (x3-x1)| where (x1,y1)=pts[i]
                # Actually need to use pts[i] as reference
                x1, y1 = pts[i]
                area = abs(dx1 * (y3 - y1) - dy1 * (x3 - x1))
                if area == target_area:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        if best_p is None:
            # Pick random unused
            for _ in range(100):
                p = rng.choice(all_coords)
                if p not in used:
                    best_p = p
                    break
            if best_p is None:
                break
        pts.append(best_p)
        used.add(best_p)
    
    # Local search: swap points to improve target-area count
    def _count_target(pts, target):
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
    
    best_pts = list(pts)
    best_cnt = _count_target(pts, target_area)
    start2 = time.time()
    while time.time() - start2 < max(0, time_limit - (time.time() - start)):
        idx = rng.randint(0, 29)
        old = pts[idx]
        # Try a random replacement
        for _ in range(5):
            new = rng.choice(all_coords)
            if new in used and new != old:
                continue
            pts[idx] = new
            cnt = _count_target(pts, target_area)
            if cnt > best_cnt:
                best_cnt = cnt
                best_pts = list(pts)
                if new != old:
                    used.discard(old)
                    used.add(new)
            else:
                pts[idx] = old
        if time.time() - start2 > time_limit - (time.time() - start):
            break
    return best_pts, best_cnt

def _local_search(points, time_limit, rng):
    start = time.time()
    best_pts = list(points)
    best_score = _calculate_score(best_pts)
    cur = list(points)
    cur_score = best_score
    all_coords = [(x,y) for x in range(-30,31) for y in range(-30,31)]
    used = set(cur)
    stuck = 0
    while time.time() - start < time_limit:
        idx = rng.randint(0, 29)
        old = cur[idx]
        for _ in range(3):
            new = rng.choice(all_coords)
            if new in used and new != old:
                continue
            cur[idx] = new
            s = _calculate_score(cur)
            if s > cur_score:
                cur_score = s
                if s > best_score:
                    best_score = s
                    best_pts = list(cur)
                    used.discard(old)
                    used.add(new)
                stuck = 0
                break
            else:
                cur[idx] = old
        else:
            stuck += 1
            if stuck > 50:
                cur = list(best_pts)
                cur_score = best_score
                used = set(cur)
                stuck = 0
    return best_pts, best_score

def construct():
    start = time.time()
    deadline = start + 55.0
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
    remaining = deadline - elapsed - 5.0
    if remaining > 10.0:
        # Try several target areas
        target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        per_target = remaining / len(target_areas)
        for ta in target_areas:
            if time.time() > deadline - 5.0:
                break
            tl = min(per_target, deadline - time.time() - 5.0)
            if tl < 1.0:
                break
            pts, cnt = _targeted_greedy(ta, tl, rng)
            # Evaluate full score
            s = _calculate_score(pts)
            if s > best_score:
                best_score = s
                best_pts = list(pts)
    
    # Phase 3: local search on best
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        pts, s = _local_search(best_pts, remaining, rng)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    return best_pts
