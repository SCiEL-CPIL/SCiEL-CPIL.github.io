"""Bootstrap spawn: model-and-optimize for 16 integer points, maximize exact-4 lines."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from fractions import Fraction
from collections import defaultdict

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2: a*x + b*y = c."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_lines(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    exact_4 = 0
    has_5 = False
    for pts in line_pts.values():
        cnt = len(pts)
        if cnt == 4:
            exact_4 += 1
        elif cnt >= 5:
            has_5 = True
    return exact_4, has_5

def _score_lines_fast(points):
    """Faster scoring using point-line incidence counting."""
    n = 16
    line_count = defaultdict(int)
    # For each pair, compute line key and count
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_count[key] += 1
    # A line with k points has C(k,2) pairs
    # We need to find k for each line
    exact_4 = 0
    has_5 = False
    for key, pair_count in line_count.items():
        # Find k such that C(k,2) = pair_count
        # C(k,2) = k*(k-1)/2
        k = 0
        for candidate_k in range(2, 17):
            if candidate_k * (candidate_k - 1) // 2 == pair_count:
                k = candidate_k
                break
        if k == 4:
            exact_4 += 1
        elif k >= 5:
            has_5 = True
    return exact_4, has_5

def _build_line_map(points):
    """Build line_pts mapping: line_key -> set of point indices."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return line_pts

def _try_insert_fourth(points, line_pts=None):
    """Try to place a point that completes exactly one 3-point line to a 4-point line."""
    n = 16
    if line_pts is None:
        line_pts = _build_line_map(points)
    
    # Find all 3-point lines
    three_lines = []
    for key, pts in line_pts.items():
        if len(pts) == 3:
            three_lines.append((key, pts))
    
    if not three_lines:
        return None
    
    # For each 3-point line, find potential 4th point
    candidates = []
    existing_set = set(tuple(p) for p in points)
    for key, pts in three_lines:
        a, b, c = key
        # Line: a*x + b*y = c
        # Find integer solutions
        if b != 0:
            # y = (c - a*x) / b
            # Try x values near existing points
            xs = [points[i][0] for i in pts]
            for x_try in range(min(xs) - 5, max(xs) + 6):
                if abs(x_try) > 1000:
                    continue
                num = c - a * x_try
                if num % b == 0:
                    y_try = num // b
                    if abs(y_try) <= 1000:
                        pt = (x_try, y_try)
                        if pt not in existing_set:
                            # Check this doesn't create 5-collinear
                            bad = False
                            for k in range(n):
                                if k in pts:
                                    continue
                                key2 = _line_key(pt, points[k])
                                if key2 is not None and key2 in line_pts and len(line_pts[key2]) >= 4:
                                    bad = True
                                    break
                            if not bad:
                                candidates.append(pt)
        else:
            # a*x = c, vertical line x = c/a
            if a != 0 and c % a == 0:
                x_val = c // a
                if abs(x_val) <= 1000:
                    ys = [points[i][1] for i in pts]
                    for y_try in range(min(ys) - 5, max(ys) + 6):
                        if abs(y_try) > 1000:
                            continue
                        pt = (x_val, y_try)
                        if pt not in existing_set:
                            bad = False
                            for k in range(n):
                                if k in pts:
                                    continue
                                key2 = _line_key(pt, points[k])
                                if key2 is not None and key2 in line_pts and len(line_pts[key2]) >= 4:
                                    bad = True
                                    break
                            if not bad:
                                candidates.append(pt)
    
    if not candidates:
        return None
    return candidates[random.randrange(len(candidates))]

def _local_search(points, deadline, rng):
    """Local search to improve the number of exact-4 lines."""
    current = [p[:] for p in points]
    current_score, h5 = _score_lines_fast(current)
    if h5:
        return current, 0
    
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 0.5
    no_improve = 0
    max_no_improve = 800
    
    # Pre-build line map for faster evaluation
    line_pts = _build_line_map(current)
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        idx = rng.randint(0, 15)
        old = current[idx][:]
        
        if rng.random() < 0.4:
            new_pt = _try_insert_fourth(current, line_pts)
            if new_pt is None:
                new_x = old[0] + rng.randint(-8, 8)
                new_y = old[1] + rng.randint(-8, 8)
                new_pt = [new_x, new_y]
            else:
                new_pt = list(new_pt)
        else:
            new_x = old[0] + rng.randint(-8, 8)
            new_y = old[1] + rng.randint(-8, 8)
            new_pt = [new_x, new_y]
        
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            continue
        if any(new_pt == p for p in current):
            continue
        
        current[idx] = new_pt
        s_new, h5_new = _score_lines_fast(current)
        
        if h5_new:
            current[idx] = old
            no_improve += 1
            continue
        
        if s_new > current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
            if s_new > current_score:
                no_improve = 0
                if s_new > best_score:
                    best_score = s_new
                    best = [p[:] for p in current]
            else:
                no_improve += 1
            current_score = s_new
        else:
            current[idx] = old
            no_improve += 1
        
        T = max(0.05, T * 0.9995)
    
    return best, best_score

def _make_perturbed_grid(rng, base, num_perturb):
    """Create a perturbed version of the base grid."""
    current = [p[:] for p in base]
    for _ in range(num_perturb):
        idx = rng.randint(0, 15)
        new_x = current[idx][0] + rng.randint(-5, 5)
        new_y = current[idx][1] + rng.randint(-5, 5)
        new_pt = [new_x, new_y]
        if any(new_pt == p for p in current):
            continue
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        current[idx] = new_pt
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin (150s limit)
    
    rng = random.Random(42)
    
    # Start with a good base: 4x4 grid has many 4-point lines
    base = [[x, y] for x in range(4) for y in range(4)]
    base_score, _ = _score_lines_fast(base)
    best_points = [p[:] for p in base]
    best_score = base_score
    
    # Multiple restarts with different initial perturbations
    restarts = 40
    for r in range(restarts):
        if time.time() > deadline - 25:
            break
        
        if r == 0:
            current = [p[:] for p in base]
        elif r % 3 == 1:
            # Light perturbation
            current = _make_perturbed_grid(rng, base, rng.randint(2, 5))
        elif r % 3 == 2:
            # Heavier perturbation
            current = _make_perturbed_grid(rng, base, rng.randint(5, 10))
        else:
            # Random points near grid
            current = []
            for _ in range(16):
                x = rng.randint(-2, 6)
                y = rng.randint(-2, 6)
                pt = [x, y]
                if pt not in current:
                    current.append(pt)
            # Fill if needed
            while len(current) < 16:
                x = rng.randint(-2, 6)
                y = rng.randint(-2, 6)
                pt = [x, y]
                if pt not in current:
                    current.append(pt)
        
        s, h5 = _score_lines_fast(current)
        if h5:
            continue
        
        opt_pts, opt_score = _local_search(current, deadline, rng)
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts
    
    # Final polish: run local search on best with more iterations
    if time.time() < deadline - 10:
        final_rng = random.Random(123)
        final_pts, final_score = _local_search(best_points, deadline, final_rng)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts
    
    # Final verification
    final_score, final_h5 = _score_lines(best_points)
    if final_h5 or final_score == 0:
        # Fallback
        best_points = [p[:] for p in base]
    
    return best_points
# EVOLVE-BLOCK-END
