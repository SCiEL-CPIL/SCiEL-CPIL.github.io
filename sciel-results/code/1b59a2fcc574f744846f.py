import time
import random
import math
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    if g != 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _score(points):
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
    count = 0
    has_five = False
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
        elif len(pts) >= 5:
            has_five = True
    if has_five:
        return -10000 + count
    return count

def _valid(points):
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
    return True

def _random_affine(base_points, rng):
    for _ in range(200):
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-3, 3)
        d = rng.randint(-3, 3)
        det = a * d - b * c
        if det == 0:
            continue
        pts = []
        for x, y in base_points:
            nx = a * x + b * y
            ny = c * x + d * y
            pts.append((nx, ny))
        tx = rng.randint(-500, 500)
        ty = rng.randint(-500, 500)
        all_ok = True
        for nx, ny in pts:
            nx2 = nx + tx
            ny2 = ny + ty
            if abs(nx2) > 1000 or abs(ny2) > 1000:
                all_ok = False
                break
        if not all_ok:
            continue
        result = [[nx + tx, ny + ty] for nx, ny in pts]
        if _valid(result):
            return result
    return None

def _local_search(points, time_limit=10.0, rng=None):
    if rng is None:
        rng = random.Random()
    best = [list(p) for p in points]
    best_score = _score(best)
    start_time = time.time()
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(16):
            if time.time() - start_time >= time_limit:
                break
            for _ in range(3):
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                new_x = best[i][0] + dx
                new_y = best[i][1] + dy
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                trial = [list(p) for p in best]
                trial[i] = [new_x, new_y]
                if len(set(tuple(p) for p in trial)) != 16:
                    continue
                s = _score(trial)
                if s > best_score:
                    best = trial
                    best_score = s
                    improved = True
                    break
        if not improved:
            for i in range(16):
                if time.time() - start_time >= time_limit:
                    break
                for _ in range(5):
                    dx = rng.randint(-20, 20)
                    dy = rng.randint(-20, 20)
                    if dx == 0 and dy == 0:
                        continue
                    new_x = best[i][0] + dx
                    new_y = best[i][1] + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    trial = [list(p) for p in best]
                    trial[i] = [new_x, new_y]
                    if len(set(tuple(p) for p in trial)) != 16:
                        continue
                    s = _score(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                        improved = True
                        break
            if not improved:
                i = rng.randint(0, 15)
                trial = [list(p) for p in best]
                trial[i] = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                if len(set(tuple(p) for p in trial)) == 16:
                    s = _score(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                    elif rng.random() < 0.1:
                        best = trial
                        best_score = s
                if time.time() - start_time >= time_limit:
                    break
    return best, best_score

def _gen_moves(cur, rng, n=16):
    moves = []
    occ = {tuple(p) for p in cur}
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(cur[i], cur[j])
            if k is not None:
                lines[k].add(i)
    for k, idxs in lines.items():
        if len(idxs) == 3:
            il = list(idxs)
            a, b = cur[il[0]], cur[il[1]]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            g = math.gcd(abs(dx), abs(dy))
            if g:
                dx //= g
                dy //= g
            for t in range(-8, 9):
                cand = [a[0] + t * dx, a[1] + t * dy]
                if abs(cand[0]) <= 1000 and abs(cand[1]) <= 1000 and tuple(cand) not in occ:
                    moves.append((il[0], cand))
    for _ in range(60):
        i = rng.randrange(n)
        p = cur[i]
        cand = [p[0] + rng.randint(-4, 4), p[1] + rng.randint(-4, 4)]
        if abs(cand[0]) <= 1000 and abs(cand[1]) <= 1000 and tuple(cand) not in occ:
            moves.append((i, cand))
    i = rng.randrange(n)
    cand = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
    if tuple(cand) not in occ:
        moves.append((i, cand))
    return moves

def _polish(cur, curescore, rng, time_left_fn):
    improved = True
    n = 16
    while improved and time_left_fn() > 2.0:
        improved = False
        bestmove = None
        bests = curescore
        occ = {tuple(p) for p in cur}
        cands = set()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    a, b = cur[j], cur[k]
                    dx = b[0] - a[0]
                    dy = b[1] - a[1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g:
                        dx //= g
                        dy //= g
                    for t in range(-3, 4):
                        cand = (a[0] + t * dx, a[1] + t * dy)
                        if abs(cand[0]) <= 1000 and abs(cand[1]) <= 1000 and cand not in occ:
                            cands.add((i, cand))
        rng.shuffle(list(cands))
        for i, cand in list(cands):
            if time_left_fn() < 2.0:
                break
            trial = [list(p) for p in cur]
            trial[i] = list(cand)
            s = _score(trial)
            if s > bests:
                bests = s
                bestmove = (i, cand)
        if bestmove:
            i, cand = bestmove
            cur = [list(p) for p in cur]
            cur[i] = list(cand)
            curescore = bests
            improved = True
    return cur, curescore

# ===== NEW: Line-based structural search =====

def _generate_candidate_lines():
    """Generate all lines with exactly 4 integer points in a small region."""
    # Use a compact region to keep candidate count manageable
    # Lines through 4 collinear integer points in [-20, 20]^2
    lines = []
    seen = set()
    for x1 in range(-20, 21):
        for y1 in range(-20, 21):
            for x2 in range(-20, 21):
                for y2 in range(-20, 21):
                    if x1 == x2 and y1 == y2:
                        continue
                    key = _line_key((x1, y1), (x2, y2))
                    if key is None or key in seen:
                        continue
                    seen.add(key)
                    dx, dy, c = key
                    # Find all integer points on this line in range
                    pts = []
                    for x in range(-20, 21):
                        # dy*x - dx*y = c  =>  y = (dy*x - c) / dx if dx != 0
                        if dx != 0:
                            num = dy * x - c
                            if num % dx == 0:
                                y = num // dx
                                if -20 <= y <= 20:
                                    pts.append((x, y))
                        else:
                            # dx == 0, vertical line: x = c/dy... wait
                            # Line: dy*x - 0*y = c => x = c/dy
                            if c % dy == 0:
                                x_val = c // dy
                                if -20 <= x_val <= 20:
                                    for y in range(-20, 21):
                                        pts.append((x_val, y))
                    if len(pts) == 4:
                        lines.append((key, pts))
    return lines

def _line_based_search(cand_lines, time_limit=30.0, rng=None):
    """Backtracking: select 16 points maximizing 4-point lines."""
    if rng is None:
        rng = random.Random()
    start_time = time.time()
    
    # Build point -> lines index
    point_lines = defaultdict(list)
    for idx, (key, pts) in enumerate(cand_lines):
        for p in pts:
            point_lines[p].append(idx)
    
    all_points = sorted(point_lines.keys())
    
    best_score = 0
    best_set = None
    
    # Greedy + local search hybrid
    # Start with random 16-point sets, then optimize
    
    def _quick_score(pt_set):
        """Fast score for a set of points."""
        line_map = defaultdict(set)
        pts = list(pt_set)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(pts[i], pts[j])
                if key is not None:
                    line_map[key].add(i)
        count = 0
        has_five = False
        for key, idxs in line_map.items():
            if len(idxs) == 4:
                count += 1
            elif len(idxs) >= 5:
                has_five = True
        if has_five:
            return -10000 + count
        return count
    
    # Strategy: build 16-point sets by greedily completing lines
    for restart in range(50):
        if time.time() - start_time >= time_limit:
            break
        
        # Random initial 16 points
        if len(all_points) < 16:
            break
        chosen = set()
        while len(chosen) < 16:
            p = rng.choice(all_points)
            chosen.add(p)
        
        cur_score = _quick_score(chosen)
        
        # Local search: swap points to improve
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try swapping each point with a candidate from line completions
            for i, p in enumerate(list(chosen)):
                if time.time() - start_time >= time_limit:
                    break
                # Find lines through p and other chosen points
                for line_idx in point_lines.get(p, []):
                    key, pts = cand_lines[line_idx]
                    # Other points on this line
                    others = [q for q in pts if q != p]
                    for q in others:
                        if q in chosen:
                            # This line has p and q; find 2 more points on line not in chosen
                            remaining = [r for r in pts if r != p and r != q and r not in chosen]
                            if len(remaining) >= 2:
                                # Try adding one of remaining, removing a random other point
                                for r in remaining[:2]:
                                    for other in list(chosen):
                                        if other == p:
                                            continue
                                        new_set = set(chosen)
                                        new_set.remove(other)
                                        new_set.add(r)
                                        if len(new_set) != 16:
                                            continue
                                        s = _quick_score(new_set)
                                        if s > cur_score:
                                            chosen = new_set
                                            cur_score = s
                                            improved = True
                                            break
                                    if improved:
                                        break
                            if improved:
                                break
                if improved:
                    break
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = set(chosen)
    
    if best_set is None:
        return None, 0
    
    # Convert to list
    result = [[x, y] for x, y in best_set]
    return result, best_score

def _structured_grid_search(time_limit=10.0, rng=None):
    """Search structured configurations: grids, lattices, etc."""
    if rng is None:
        rng = random.Random()
    best_pts = None
    best_score = -1
    start_time = time.time()
    
    # Try various 4x4 grid variants with different spacings
    for sx in range(1, 30):
        if time.time() - start_time >= time_limit:
            break
        for sy in range(1, 30):
            if time.time() - start_time >= time_limit:
                break
            pts = [[i * sx, j * sy] for i in range(4) for j in range(4)]
            if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in pts):
                continue
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = [list(p) for p in pts]
    
    # Try sheared grids
    for sx in range(1, 20):
        if time.time() - start_time >= time_limit:
            break
        for sy in range(1, 20):
            if time.time() - start_time >= time_limit:
                break
            for shear in range(-3, 4):
                pts = [[i * sx + shear * j, j * sy] for i in range(4) for j in range(4)]
                if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in pts):
                    continue
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = [list(p) for p in pts]
    
    # Try 2x8 and other rectangular grids
    for shape in [(2, 8), (8, 2), (4, 4)]:
        if time.time() - start_time >= time_limit:
            break
        for sx in range(1, 30):
            if time.time() - start_time >= time_limit:
                break
            for sy in range(1, 30):
                if time.time() - start_time >= time_limit:
                    break
                pts = [[i * sx, j * sy] for i in range(shape[0]) for j in range(shape[1])]
                if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in pts):
                    continue
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = [list(p) for p in pts]
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    rng = random.Random(20240517)
    
    def time_left():
        return deadline - time.time()
    
    # Start with 4x4 grid
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    best_pts = [list(p) for p in base_grid]
    best_score = _score(best_pts)
    
    # Try affine transforms
    for _ in range(300):
        if time.time() > deadline - 20:
            break
        transformed = _random_affine(base_grid, rng)
        if transformed is not None:
            s = _score(transformed)
            if s > best_score:
                best_score = s
                best_pts = transformed
    
    # Structured grid search
    if time_left() > 15:
        pts, s = _structured_grid_search(time_limit=min(10.0, time_left() - 5), rng=rng)
        if pts is not None and s > best_score:
            best_score = s
            best_pts = pts
    
    # Line-based search (new structural approach)
    if time_left() > 20:
        cand_lines = _generate_candidate_lines()
        if cand_lines:
            pts, s = _line_based_search(cand_lines, time_limit=min(30.0, time_left() - 10), rng=rng)
            if pts is not None and s > best_score:
                best_score = s
                best_pts = pts
    
    # Local search from best
    if time_left() > 10:
        pts, s = _local_search(best_pts, time_limit=min(15.0, time_left() - 3), rng=rng)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Polish
    if time_left() > 5:
        best_pts, best_score = _polish(best_pts, best_score, rng, time_left)
    
    # Final local search
    if time_left() > 5:
        pts, s = _local_search(best_pts, time_limit=min(8.0, time_left() - 2), rng=rng)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Final check
    if not _valid(best_pts):
        best_pts = [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts

# EVOLVE-BLOCK-END
