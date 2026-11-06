import time
import random
import sys
from math import gcd
from itertools import combinations

DEADLINE = 110  # Safety margin before 120s kill

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    a = -dy
    b = dx
    c = dx * y1 - dy * x1
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    g2 = gcd(abs(a), abs(b))
    if g2 > 1:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _build_lines(points):
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            s = line_map.get(key)
            if s is None:
                s = set()
                line_map[key] = s
            s.add(i)
            s.add(j)
    lines = []
    line_masks = []
    for s in line_map.values():
        if len(s) >= 3:
            lines.append(frozenset(s))
            m = 0
            for idx in s:
                m |= 1 << idx
            line_masks.append(m)
    return lines, len(lines), line_masks

def _check_no_four(points, lines):
    for L in lines:
        if len(L) >= 4:
            return False
    return True

def _count_triples(lines):
    """Count number of collinear triples (lines with exactly 3 points)."""
    return sum(1 for L in lines if len(L) == 3)

def _greedy_construct(rng, max_iter=5):
    """
    Greedily construct 20 points, choosing each new point to minimize
    the number of new collinear triples created.
    """
    best_pts = None
    best_triple_count = 999
    
    for attempt in range(max_iter):
        points = []
        # Start with a small seed set
        seed_size = 3
        for _ in range(seed_size):
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            points.append([x, y])
        
        # Greedily add points
        for step in range(20 - seed_size):
            best_candidate = None
            best_new_triples = 999
            best_candidate_pos = None
            
            # Try many random candidates
            for trial in range(200):
                cx = rng.randint(-100, 100)
                cy = rng.randint(-100, 100)
                cand = [cx, cy]
                
                # Check distinctness
                if any(cand[0] == p[0] and cand[1] == p[1] for p in points):
                    continue
                
                # Count new collinear triples created by adding this point
                new_triples = 0
                temp_points = points + [cand]
                
                # For each pair of existing points, check if cand is collinear
                n = len(points)
                for i in range(n):
                    for j in range(i + 1, n):
                        key = _line_key(points[i], points[j])
                        if key is None:
                            continue
                        # Check if cand is on this line
                        a, b, c = key
                        if a * cand[0] + b * cand[1] + c == 0:
                            # This line now has 3 points: i, j, cand
                            new_triples += 1
                
                if new_triples < best_new_triples:
                    best_new_triples = new_triples
                    best_candidate = cand
                    best_candidate_pos = (cx, cy)
                
                if new_triples == 0:
                    break
            
            if best_candidate is None:
                # Fallback: random
                best_candidate = [rng.randint(-100, 100), rng.randint(-100, 100)]
            
            points.append(best_candidate)
        
        # Evaluate final configuration
        lines, lc, lm = _build_lines(points)
        if not _check_no_four(points, lines):
            continue
        
        tc = _count_triples(lines)
        if tc < best_triple_count:
            best_triple_count = tc
            best_pts = [p[:] for p in points]
    
    return best_pts, best_triple_count

def _break_triples(points, time_budget, rng):
    """
    Targeted repair: find lines with exactly 3 points, and try to move
    one of the three points to a location that breaks the triple without
    creating new 4-collinearities.
    """
    start = time.time()
    cur = [p[:] for p in points]
    
    while time.time() - start < time_budget:
        lines, lc, lm = _build_lines(cur)
        if not _check_no_four(cur, lines):
            break
        
        # Find lines with exactly 3 points
        triples = [L for L in lines if len(L) == 3]
        if not triples:
            break
        
        # Pick a random triple
        triple = triples[rng.randrange(len(triples))]
        i, j, k = list(triple)
        
        # Try moving one of the three points
        moved = False
        for idx in [i, j, k]:
            ox, oy = cur[idx][:]
            
            for trial in range(50):
                nx = rng.randint(-100, 100)
                ny = rng.randint(-100, 100)
                
                if nx == ox and ny == oy:
                    continue
                if any(cur[m][0] == nx and cur[m][1] == ny for m in range(20) if m != idx):
                    continue
                
                cur[idx] = [nx, ny]
                
                # Check no 4-collinear
                lines2, lc2, lm2 = _build_lines(cur)
                if not _check_no_four(cur, lines2):
                    cur[idx] = [ox, oy]
                    continue
                
                # Count triples
                tc2 = _count_triples(lines2)
                tc1 = _count_triples(lines)
                
                if tc2 < tc1:
                    moved = True
                    break
                else:
                    cur[idx] = [ox, oy]
            
            if moved:
                break
        
        if not moved:
            # Stuck, try a different approach: move a random point
            idx = rng.randrange(20)
            ox, oy = cur[idx][:]
            for trial in range(30):
                nx = rng.randint(-100, 100)
                ny = rng.randint(-100, 100)
                if nx == ox and ny == oy:
                    continue
                if any(cur[m][0] == nx and cur[m][1] == ny for m in range(20) if m != idx):
                    continue
                
                cur[idx] = [nx, ny]
                lines2, lc2, lm2 = _build_lines(cur)
                if not _check_no_four(cur, lines2):
                    cur[idx] = [ox, oy]
                    continue
                
                tc2 = _count_triples(lines2)
                tc1 = _count_triples(lines)
                
                if tc2 <= tc1:
                    moved = True
                    break
                else:
                    cur[idx] = [ox, oy]
            
            if not moved:
                cur[idx] = [ox, oy]
                break
    
    return cur

def _max_cap_free(points, lines, line_masks, line_count, time_limit=2.0):
    """Exact max 3-collinear-free subset via branch and bound."""
    n = len(points)
    start = time.time()
    timed_out = [False]

    # For each point, the lines it belongs to (by line index)
    point_lines = [[] for _ in range(n)]
    for li, L in enumerate(lines):
        for idx in L:
            point_lines[idx].append(li)

    # Line info: how many points are colored 1 on each line
    line_ones = [0] * line_count
    
    color = [0] * n
    best = [0]
    cur = [0]

    def bound(idx):
        return cur[0] + (n - idx)

    def dfs(idx):
        if timed_out[0]:
            return
        if time.time() - start > time_limit:
            timed_out[0] = True
            return
        if idx == n:
            if cur[0] > best[0]:
                best[0] = cur[0]
            return
        if bound(idx) <= best[0]:
            return
        
        p = idx
        # Try in (color 1)
        ok = True
        touched = []
        for li in point_lines[p]:
            if line_ones[li] >= 2:
                ok = False
                break
            line_ones[li] += 1
            touched.append(li)
        if ok:
            color[p] = 1
            cur[0] += 1
            dfs(idx + 1)
            cur[0] -= 1
            for li in touched:
                line_ones[li] -= 1
            color[p] = 0
        if timed_out[0]:
            return
        # Try out (color 0)
        color[p] = 0
        dfs(idx + 1)

    dfs(0)
    return best[0]

def _valid(points):
    if len(set(tuple(p) for p in points)) != 20:
        return False
    if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in points):
        return False
    lines, lc, lm = _build_lines(points)
    return _check_no_four(points, lines)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_points = None
    best_score = 999
    
    # Phase 1: Greedy construction with multiple seeds
    for seed in range(30):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        r = random.Random(seed)
        pts, tc = _greedy_construct(r, max_iter=3)
        
        if pts is None:
            continue
        
        if not _valid(pts):
            continue
        
        lines, lc, lm = _build_lines(pts)
        s = _max_cap_free(pts, lines, lm, lc, 1.0)
        
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            
            # If good, try to improve
            if s <= 16 and time.time() - start_time < DEADLINE - 15:
                remaining = DEADLINE - 15 - (time.time() - start_time)
                if remaining > 3:
                    refined = _break_triples(pts, min(10.0, remaining), rng)
                    if _valid(refined):
                        lines_r, lc_r, lm_r = _build_lines(refined)
                        sr = _max_cap_free(refined, lines_r, lm_r, lc_r, 1.0)
                        if sr < best_score:
                            best_score = sr
                            best_points = [p[:] for p in refined]
    
    # Phase 2: If we have a good candidate, do more targeted search
    if best_points is not None and best_score <= 15 and time.time() - start_time < DEADLINE - 20:
        remaining = DEADLINE - 20 - (time.time() - start_time)
        if remaining > 5:
            refined = _break_triples(best_points, min(20.0, remaining), rng)
            if _valid(refined):
                lines_r, lc_r, lm_r = _build_lines(refined)
                sr = _max_cap_free(refined, lines_r, lm_r, lc_r, 1.5)
                if sr < best_score:
                    best_score = sr
                    best_points = [p[:] for p in refined]
    
    # Fallback
    if best_points is None or best_score >= 20:
        # Use a parabola-based construction which has no 3 collinear
        best_points = []
        for x in range(-10, 10):
            y = x * x
            best_points.append([x, y])
        # Verify
        if not _valid(best_points):
            # Absolute fallback: simple grid with perturbations
            r = random.Random(999)
            best_points = []
            for i in range(20):
                best_points.append([r.randint(-100, 100), r.randint(-100, 100)])
    
    return best_points
