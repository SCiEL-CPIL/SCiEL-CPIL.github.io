import random
import time
import math
from itertools import combinations
from math import gcd

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            # Normalize: make (A,B) lexicographically positive, then C
            if A < 0 or (A == 0 and B < 0):
                A, B, C = -A, -B, -C
            elif A == 0 and B == 0:
                if C < 0:
                    C = -C
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _all_collinear(points):
    if len(points) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if cross != 0:
        return False
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return False
    return True

def _boroczky_conic_line():
    """
    Böröczky configuration: 6 points on a conic + 6 points on a line.
    
    The key insight: place 6 points on the parabola y = x^2 and 6 points
    on a horizontal line far away. By choosing the x-coordinates of the
    parabola points carefully, we can ensure that lines through pairs of
    parabola points intersect the far line at points that coincide with
    the 6 far-line points, creating many 3-point collinearities.
    
    For the parabola y = x^2, the line through points (a, a^2) and (b, b^2)
    has equation y = (a+b)x - ab. It intersects y = H at x = (ab + H)/(a+b).
    
    We want 6 parabola points and 6 line points such that many of these
    intersections coincide with the line points.
    
    Using a symmetric set: parabola points at x = {-5,-3,-1,1,3,5}
    and line at y = 100 with points at specific x values.
    
    Actually, let's use a simpler verified approach: the known Böröczky
    construction uses points on a conic where the "points at infinity"
    are replaced by a line at a large distance.
    
    Let me try: parabola points at x in {-2,-1,0,1,2,3} and line y=6
    with points chosen to create collinearities.
    """
    # Parabola: y = x^2
    para_xs = [-5, -3, -1, 1, 3, 5]
    para_pts = [[x, x * x] for x in para_xs]
    
    # Line: y = 100
    # Compute where lines through pairs of parabola points hit y=100
    # Line through (a,a^2) and (b,b^2): y = (a+b)x - ab
    # At y=100: x = (ab + 100) / (a+b)
    
    # We'll place 6 points on y=100 to maximize collinearities
    # Try a few candidate sets
    H = 100
    best_line_pts = None
    best_score = float('inf')
    
    # Candidate x-values for the line points
    # Lines through parabola pairs give x-intercepts:
    # For a,b in para_xs, a != b: x_int = (a*b + H) / (a+b)
    # Let's collect all possible x-intercepts
    x_intercepts = set()
    for i in range(len(para_xs)):
        for j in range(i + 1, len(para_xs)):
            a, b = para_xs[i], para_xs[j]
            if a + b != 0:
                num = a * b + H
                den = a + b
                # Only keep integer intercepts
                if num % den == 0:
                    x_intercepts.add(num // den)
    
    # Sort and pick 6 that maximize collinearities
    x_intercepts_sorted = sorted(x_intercepts)
    
    # Try all combinations of 6 from the intercepts (or from a broader set)
    # If not enough, add some nearby integers
    all_candidates = list(x_intercepts_sorted)
    # Add some integers around the intercepts
    extra = set()
    for xi in x_intercepts:
        for d in range(-3, 4):
            extra.add(xi + d)
    all_candidates = sorted(set(all_candidates) | set(x_intercepts))
    
    if len(all_candidates) < 6:
        all_candidates = list(range(-20, 21))
    
    # Try subsets of size 6
    from itertools import combinations as comb
    best_combo = None
    best_combo_score = float('inf')
    
    # Limit search to reasonable range
    limited = [x for x in all_candidates if -100 <= x <= 100]
    if len(limited) < 6:
        limited = all_candidates
    
    # Try first few combinations for speed
    for line_xs in comb(limited[:30], 6):
        line_pts = [[x, H] for x in line_xs]
        pts = para_pts + line_pts
        # Check distinctness
        seen = set()
        valid = True
        for p in pts:
            k = (p[0], p[1])
            if k in seen:
                valid = False
                break
            seen.add(k)
        if not valid:
            continue
        if _all_collinear(pts):
            continue
        s = _count_ordinary_lines(pts)
        if s < best_combo_score:
            best_combo_score = s
            best_combo = list(line_xs)
            if s <= 6:
                break
    
    if best_combo is None:
        # Fallback: use evenly spaced points
        best_combo = [0, 1, 2, 3, 4, 5]
    
    line_pts = [[x, H] for x in best_combo]
    return para_pts + line_pts

def _boroczky_symmetric():
    """
    Another Böröczky-type construction using a circle.
    
    6 points on a circle + 6 points on a line.
    Circle: x^2 + y^2 = R^2
    Line: y = M (far away)
    """
    R = 10
    # 6 points on circle x^2 + y^2 = 100
    # (0,10), (6,8), (10,0), (8,6), (0,-10), (-6,-8)
    circle_pts = [[0, 10], [6, 8], [10, 0], [8, 6], [0, -10], [-6, -8]]
    
    M = 50
    # Line through two circle points intersects y=M
    # For points (x1,y1), (x2,y2) on circle, line intersects y=M at:
    # x = x1 + (M-y1)*(x2-x1)/(y2-y1)
    
    x_intercepts = set()
    n = len(circle_pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = circle_pts[i]
            x2, y2 = circle_pts[j]
            if y2 != y1:
                num = x1 * (y2 - y1) + (M - y1) * (x2 - x1)
                den = y2 - y1
                if num % den == 0:
                    x_intercepts.add(num // den)
    
    all_candidates = sorted(x_intercepts)
    if len(all_candidates) < 6:
        for xi in x_intercepts:
            for d in range(-2, 3):
                all_candidates.append(xi + d)
        all_candidates = sorted(set(all_candidates))
    if len(all_candidates) < 6:
        all_candidates = list(range(-10, 11))
    
    from itertools import combinations as comb
    best_combo = None
    best_combo_score = float('inf')
    
    limited = [x for x in all_candidates if -50 <= x <= 50]
    if len(limited) < 6:
        limited = all_candidates
    
    for line_xs in comb(limited[:25], 6):
        line_pts = [[x, M] for x in line_xs]
        pts = circle_pts + line_pts
        seen = set()
        valid = True
        for p in pts:
            k = (p[0], p[1])
            if k in seen:
                valid = False
                break
            seen.add(k)
        if not valid:
            continue
        if _all_collinear(pts):
            continue
        s = _count_ordinary_lines(pts)
        if s < best_combo_score:
            best_combo_score = s
            best_combo = list(line_xs)
            if s <= 6:
                break
    
    if best_combo is None:
        best_combo = [0, 1, 2, 3, 4, 5]
    
    line_pts = [[x, M] for x in best_combo]
    return circle_pts + line_pts

def _grid_3x4():
    """3x4 grid."""
    pts = []
    for i in range(3):
        for j in range(4):
            pts.append([j, i])
    return pts

def _grid_2x6():
    """2x6 grid."""
    pts = []
    for i in range(2):
        for j in range(6):
            pts.append([j, i])
    return pts

def _targeted_local_search(points, time_limit=40):
    """
    Targeted local search: try to create new collinearities by moving points
    to positions that lie on lines through pairs of other points.
    """
    start = time.time()
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    
    if current_score <= 6:
        return current, current_score
    
    best = [p[:] for p in current]
    best_score = current_score
    
    # Precompute lines through pairs
    def get_lines_through_pair(pts, exclude_idx):
        """Get all integer points on lines through pairs of pts (excluding exclude_idx)."""
        n = len(pts)
        candidates = set()
        for i in range(n):
            if i == exclude_idx:
                continue
            for j in range(i + 1, n):
                if j == exclude_idx:
                    continue
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                # Points on this line: (x1 + t*dx, y1 + t*dy) for integer t
                for t in range(-50, 51):
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        candidates.add((nx, ny))
        return candidates
    
    for step in range(20000):
        if time.time() - start > time_limit:
            break
        if best_score <= 6:
            break
        
        idx = random.randint(0, 11)
        
        # Get candidate positions on lines through pairs of other points
        candidates = get_lines_through_pair(current, idx)
        
        # Also add some random nearby positions
        for _ in range(5):
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            nx = current[idx][0] + dx
            ny = current[idx][1] + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                candidates.add((nx, ny))
        
        # Remove current position
        candidates.discard((current[idx][0], current[idx][1]))
        
        if not candidates:
            continue
        
        # Try a random subset of candidates
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        improved = False
        for nx, ny in cand_list[:20]:  # Try up to 20 candidates
            if time.time() - start > time_limit:
                break
            
            trial = [p[:] for p in current]
            trial[idx] = [nx, ny]
            
            # Check distinctness
            seen = set()
            valid = True
            for p in trial:
                k = (p[0], p[1])
                if k in seen:
                    valid = False
                    break
                seen.add(k)
            if not valid:
                continue
            
            if _all_collinear(trial):
                continue
            
            trial_score = _count_ordinary_lines(trial)
            if trial_score < current_score:
                current = trial
                current_score = trial_score
                improved = True
                if current_score < best_score:
                    best = [p[:] for p in current]
                    best_score = current_score
                if best_score <= 6:
                    return best, best_score
                break
            elif trial_score < current_score + 2 and random.random() < 0.05:
                current = trial
                current_score = trial_score
        
        if not improved and random.random() < 0.02:
            # Occasional random restart of one point
            idx2 = random.randint(0, 11)
            trial = [p[:] for p in current]
            trial[idx2] = [random.randint(-100, 100), random.randint(-100, 100)]
            seen = set()
            valid = True
            for p in trial:
                k = (p[0], p[1])
                if k in seen:
                    valid = False
                    break
                seen.add(k)
            if valid and not _all_collinear(trial):
                trial_score = _count_ordinary_lines(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    if current_score < best_score:
                        best = [p[:] for p in current]
                        best_score = current_score
    
    return best, best_score

def construct():
    start = time.time()
    deadline = start + 145  # Safety margin before 150s kill
    
    # Try deterministic constructions first
    candidates = []
    
    # 1. Böröczky conic + line (parabola)
    try:
        bl_para = _boroczky_conic_line()
        score_bl_para = _count_ordinary_lines(bl_para)
        candidates.append((score_bl_para, bl_para))
    except Exception:
        pass
    
    # 2. Böröczky circle + line
    try:
        bl_circ = _boroczky_symmetric()
        score_bl_circ = _count_ordinary_lines(bl_circ)
        candidates.append((score_bl_circ, bl_circ))
    except Exception:
        pass
    
    # 3. 3x4 grid
    grid_3x4 = _grid_3x4()
    score_3x4 = _count_ordinary_lines(grid_3x4)
    candidates.append((score_3x4, grid_3x4))
    
    # 4. 2x6 grid
    grid_2x6 = _grid_2x6()
    score_2x6 = _count_ordinary_lines(grid_2x6)
    candidates.append((score_2x6, grid_2x6))
    
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    best_score, best_points = candidates[0]
    
    # If we already have 6, return immediately
    if best_score <= 6:
        return best_points
    
    # Use targeted local search from the best candidate
    if time.time() < deadline - 5:
        remaining_time = min(50, deadline - time.time() - 5)
        search_points, search_score = _targeted_local_search(best_points, time_limit=remaining_time)
        if search_score < best_score:
            best_score = search_score
            best_points = search_points
    
    # Also try local search from other candidates if time permits
    for score, pts in candidates[1:]:
        if time.time() > deadline - 10:
            break
        if score >= best_score:
            break
        remaining_time = min(20, deadline - time.time() - 10)
        if remaining_time <= 5:
            break
        search_points, search_score = _targeted_local_search(pts, time_limit=remaining_time)
        if search_score < best_score:
            best_score = search_score
            best_points = search_points
    
    if time.time() > deadline - 2:
        return best_points
    
    # Final verification
    final_score = _count_ordinary_lines(best_points)
    if final_score < best_score:
        best_score = final_score
    
    return best_points
