"""Rewritten search: Simulated Annealing + Smart Moves to maximize exact-4 lines."""
import time
import random
from math import gcd
from collections import defaultdict

# EVOLVE-BLOCK-START

_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)


def _line_key(a, b):
    """Canonical key for the line through integer points a=(x1,y1), b=(x2,y2).
    Returns (A,B,C) with A*x+B*y=C, gcd(A,B)=1, and a sign convention."""
    x1, y1 = a
    x2, y2 = b
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g
    # sign convention: make first non-zero of (A,B) positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _score_and_lines(points):
    """Return (num_exact4_lines, num_lines_with_5plus, line_map).
    line_map maps line_key -> list of indices on that line (len>=3)."""
    n = len(points)
    groups = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            groups[k].append((i, j))
    # Build line -> set of points
    line_pts = defaultdict(set)
    for k, pairs in groups.items():
        for (i, j) in pairs:
            line_pts[k].add(i)
            line_pts[k].add(j)
    exact4 = 0
    fiveplus = 0
    for k, s in line_pts.items():
        cnt = len(s)
        if cnt == 4:
            exact4 += 1
        elif cnt >= 5:
            fiveplus += 1
    return exact4, fiveplus, line_pts


def _valid(points):
    """No five collinear, all distinct, |x|,|y|<=1000."""
    if len(set(map(tuple, points))) != len(points):
        return False
    for (x, y) in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, fiveplus, _ = _score_and_lines(points)
    return fiveplus == 0


def _score(points):
    exact4, fiveplus, _ = _score_and_lines(points)
    if fiveplus > 0:
        return -10**9 - fiveplus
    return exact4


def _get_smart_moves(idx, points, line_pts, rng):
    """Generate candidate moves for point idx based on current geometry.
    Strategy:
    1. If point is on a 5+ line, move it off.
    2. Try to complete a 4-line: find a line with 3 other points, move to 4th spot.
    3. Random small moves.
    """
    cands = set()
    x0, y0 = points[idx]
    
    # 1. Avoid 5+ lines
    # Check if idx is in any 5+ line
    for k, s in line_pts.items():
        if idx in s and len(s) >= 5:
            # Try to move to a random nearby point not on this line
            for _ in range(5):
                nx = x0 + rng.randint(-5, 5)
                ny = y0 + rng.randint(-5, 5)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    # Check if this new point is on the same 5+ line
                    # We need to check if (nx, ny) satisfies A*nx + B*ny = C for the line key
                    A, B, C = k
                    if A*nx + B*ny != C:
                        cands.add((nx, ny))
                    else:
                        # Try slightly further
                        nx = x0 + rng.randint(-10, 10)
                        ny = y0 + rng.randint(-10, 10)
                        if abs(nx) <= 1000 and abs(ny) <= 1000:
                             if A*nx + B*ny != C:
                                cands.add((nx, ny))

    # 2. Try to complete 4-lines
    # Find lines with exactly 3 points that do NOT include idx
    # If we move idx to the 4th point of such a line, we gain a 4-line.
    # However, we must ensure we don't create a 5-line or break other 4-lines.
    # This is complex to evaluate on the fly, so we just generate candidates on lines with 3 points.
    for k, s in line_pts.items():
        if idx not in s and len(s) == 3:
            # The line is determined by the 3 points.
            # We want to move idx to a point on this line.
            # Pick two points from s to define the line direction.
            p1_idx, p2_idx = list(s)[0], list(s)[1]
            p1 = points[p1_idx]
            p2 = points[p2_idx]
            # Direction vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Generate points on this line near the existing 3
            # We can just pick a random integer t and compute p1 + t*(dx, dy)
            # But we need integer coordinates.
            # Let's just try a few integer steps along the line
            for t in range(-3, 4):
                nx = p1[0] + t * dx
                ny = p1[1] + t * dy
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    # Ensure it's not one of the existing points
                    if (nx, ny) not in [tuple(points[i]) for i in s]:
                        cands.add((nx, ny))

    # 3. Random moves
    for _ in range(20):
        dx = rng.randint(-8, 8)
        dy = rng.randint(-8, 8)
        nx, ny = x0 + dx, y0 + dy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            cands.add((nx, ny))
            
    return list(cands)


def _simulated_annealing(points, rng, deadline_fn):
    points = [list(p) for p in points]
    cur_score = _score(points)
    cur_line_pts = None
    # Initial score calculation to get line_pts
    _, _, cur_line_pts = _score_and_lines(points)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    improved = True
    best_points = [list(p) for p in points]
    best_score = cur_score
    
    while deadline_fn() > 5.0 and T > T_min:
        improved = False
        n = len(points)
        order = list(range(n))
        rng.shuffle(order)
        
        for idx in order:
            if deadline_fn() < 5.0:
                return best_points, best_score
            
            moves = _get_smart_moves(idx, points, cur_line_pts, rng)
            if not moves:
                continue
                
            # Pick a random move from candidates
            # To keep it fast, we don't evaluate all, we pick one or a few
            # Let's pick one random move
            move = rng.choice(moves)
            
            # Check distinctness
            if any((p[0] == move[0] and p[1] == move[1]) for j, p in enumerate(points) if j != idx):
                continue
                
            old = points[idx]
            points[idx] = list(move)
            
            new_score, new_fiveplus, new_line_pts = _score_and_lines(points)
            
            # Handle penalty for 5+ lines in score comparison
            # _score returns negative large number for invalid
            s_val = _score(points)
            
            delta = s_val - cur_score
            
            if delta > 0:
                # Accept
                cur_score = s_val
                cur_line_pts = new_line_pts
                improved = True
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = [list(p) for p in points]
            else:
                # Metropolis criterion
                if rng.random() < pow(2.718281828, delta / T):
                    cur_score = s_val
                    cur_line_pts = new_line_pts
                else:
                    # Reject
                    points[idx] = old
                    
        T *= cooling
        
    return best_points, best_score


def _seed_grid(step, ox, oy, s=4):
    return [[ox + i * step, oy + j * step] for i in range(s) for j in range(s)]


def _seed_affine_grid(rng):
    base = [[i, j] for i in range(4) for j in range(4)]
    while True:
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-3, 3)
        d = rng.randint(-3, 3)
        det = a * d - b * c
        if det != 0:
            break
    ox = rng.randint(-50, 50)
    oy = rng.randint(-50, 50)
    pts = []
    for (x, y) in base:
        nx = a * x + b * y + ox
        ny = c * x + d * y + oy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            pts.append([nx, ny])
    return pts


def _seed_perturbed_grid(rng):
    base = _seed_grid(1, 0, 0)
    pts = []
    for (x, y) in base:
        dx = rng.randint(-3, 3)
        dy = rng.randint(-3, 3)
        pts.append([x + dx, y + dy])
    return pts


def _seed_random(rng):
    pts = []
    for _ in range(16):
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        pts.append([x, y])
    return pts


def construct():
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = _START + 118.0
    rng = random.Random(12345)

    seeds = []
    # Canonical grid
    seeds.append(_seed_grid(1, 0, 0))
    # Various affine
    for _ in range(50):
        s = _seed_affine_grid(rng)
        if len(s) == 16:
            seeds.append(s)
    # Perturbed
    for _ in range(50):
        seeds.append(_seed_perturbed_grid(rng))
    # Random
    for _ in range(20):
        seeds.append(_seed_random(rng))

    best_points = None
    best_score = -10**12

    # Run SA on multiple seeds
    # We don't have time for many long runs, so we run short runs on many seeds
    # Or we run one long run on the best seed?
    # Let's try a mix: quick evaluation of seeds, then SA on top few.
    
    scored_seeds = []
    for s in seeds:
        if len(set(map(tuple, s))) != 16:
            continue
        if any(abs(x) > 1000 or abs(y) > 1000 for (x, y) in s):
            continue
        sc = _score(s)
        scored_seeds.append((sc, s))
        
    # Sort by score
    scored_seeds.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 10 seeds and run SA on them
    for sc, s in scored_seeds[:10]:
        if _remaining() < 10.0:
            break
            
        # Run SA
        # We need to pass a deadline function that stops earlier if global time is low
        # But SA uses its own internal loop.
        # Let's just run it.
        pts, sc2 = _simulated_annealing(s, rng, _remaining)
        if sc2 > best_score:
            best_score = sc2
            best_points = pts

    # If we still have time, run a final SA on the best found
    if best_points and _remaining() > 20.0:
        pts, sc2 = _simulated_annealing(best_points, rng, _remaining)
        if sc2 > best_score:
            best_score = sc2
            best_points = pts

    # Final safety
    if best_points is None or not _valid(best_points):
        best_points = [[x, y] for x in range(4) for y in range(4)]

    return [[int(x), int(y)] for (x, y) in best_points]

# EVOLVE-BLOCK-END
