"""Explicit construction for the 12-point, few-ordinary-lines task.

Strategy:
  - Structured seeds (grids, two-line, perturbed).
  - Simulated annealing with line-intersection candidate generation.
  - Local search with intersection-based moves.
  - Deterministic polish pass at the end.
  - Deadline guard for safety.
"""
import time
import random
import math
from math import gcd

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150 s kill


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)


def _intersection_candidates(points, exclude_idx, max_cands=200):
    """Generate candidate integer points at intersections of lines through pairs of points.
    
    For each pair of points (i,j) with i,j != exclude_idx, compute the line.
    For each pair of such lines, compute their intersection. If integer and valid,
    it's a candidate.
    """
    n = len(points)
    # Collect lines through pairs of points (excluding the point being moved)
    lines = []
    for i in range(n):
        if i == exclude_idx:
            continue
        x1, y1 = points[i]
        for j in range(i + 1, n):
            if j == exclude_idx:
                continue
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            lines.append((A, B, C))
    
    # Remove duplicate lines
    unique_lines = list(set(lines))
    
    # Compute intersections of pairs of lines
    cands = set()
    for i in range(len(unique_lines)):
        A1, B1, C1 = unique_lines[i]
        for j in range(i + 1, len(unique_lines)):
            A2, B2, C2 = unique_lines[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue  # parallel
            x_num = B1 * C2 - B2 * C1
            y_num = A2 * C1 - A1 * C2
            if x_num % det != 0 or y_num % det != 0:
                continue
            x = x_num // det
            y = y_num // det
            if abs(x) > 10**9 or abs(y) > 10**9:
                continue
            cands.add((x, y))
            if len(cands) >= max_cands:
                break
        if len(cands) >= max_cands:
            break
    
    return list(cands)


def _line_extension_candidates(points, exclude_idx, max_cands=100):
    """Generate candidates by extending lines through pairs of points."""
    n = len(points)
    cands = set()
    for i in range(n):
        if i == exclude_idx:
            continue
        x1, y1 = points[i]
        for j in range(i + 1, n):
            if j == exclude_idx:
                continue
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-3, 4):
                if k == 0:
                    continue
                nx = x1 + k * dx
                ny = y1 + k * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    cands.add((nx, ny))
                if len(cands) >= max_cands:
                    return list(cands)
    return list(cands)


def _small_perturbation_candidates(old, radius=4):
    """Small random perturbations around a point."""
    cands = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            cands.add((old[0] + dx, old[1] + dy))
    return list(cands)


def _simulated_annealing(points, max_iters=300000, seed=7):
    """Simulated annealing with structured candidate generation."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    rng = random.Random(seed)
    T = 2.0
    T_min = 0.01
    cooling = 0.99998
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        if T < T_min:
            T = T_min
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        
        # Choose candidate generation strategy
        r = rng.random()
        if r < 0.4:
            # Intersection-based candidates
            cands = _intersection_candidates(cur, idx, max_cands=100)
        elif r < 0.7:
            # Line extension
            cands = _line_extension_candidates(cur, idx, max_cands=80)
        else:
            # Small perturbation
            cands = _small_perturbation_candidates(old, radius=3)
        
        # Filter and pick a candidate
        valid_cands = [c for c in cands if c not in cur_set and abs(c[0]) <= 10**9 and abs(c[1]) <= 10**9]
        if not valid_cands:
            continue
        
        cand = rng.choice(valid_cands)
        
        new_pts = [p[:] for p in cur]
        new_pts[idx] = list(cand)
        if not _valid(new_pts):
            continue
        
        new_score = _count_ordinary(new_pts)
        delta = new_score - cur_score
        
        if delta < 0 or (rng.random() < math.exp(-delta / T) if delta > 0 else 1.0):
            cur = new_pts
            cur_score = new_score
            cur_set = set(tuple(p) for p in cur)
            if cur_score < best_score:
                best_score = cur_score
                best = [p[:] for p in cur]
        
        T *= cooling
    
    return best, best_score


def _local_search(points, max_iters=200000, seed=123):
    """Aggressive local search with intersection-based moves."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(seed)
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        improved = False
        
        # Strategy 1: Intersection-based candidates
        cands = _intersection_candidates(cur, idx, max_cands=150)
        valid_cands = [c for c in cands if c not in cur_set and abs(c[0]) <= 10**9 and abs(c[1]) <= 10**9]
        rng.shuffle(valid_cands)
        for cand in valid_cands[:30]:
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 2: Line extension
        cands = _line_extension_candidates(cur, idx, max_cands=80)
        valid_cands = [c for c in cands if c not in cur_set and abs(c[0]) <= 10**9 and abs(c[1]) <= 10**9]
        rng.shuffle(valid_cands)
        for cand in valid_cands[:20]:
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 3: Small perturbation
        cands = _small_perturbation_candidates(old, radius=5)
        valid_cands = [c for c in cands if c not in cur_set and abs(c[0]) <= 10**9 and abs(c[1]) <= 10**9]
        rng.shuffle(valid_cands)
        for cand in valid_cands[:15]:
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
    
    return cur, cur_score


def _deterministic_polish(points):
    """Final deterministic pass: try all intersection candidates for each point."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    
    for idx in range(12):
        cands = _intersection_candidates(cur, idx, max_cands=200)
        valid_cands = [c for c in cands if tuple(c) != tuple(cur[idx]) and abs(c[0]) <= 10**9 and abs(c[1]) <= 10**9]
        for cand in valid_cands:
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                break  # restart with improved point
    
    return cur, cur_score


def _make_seed_grid(rows=3, cols=4, ox=0, oy=0, scale=1):
    """rows x cols grid with optional offset and scale."""
    return [[(x * scale + ox), (y * scale + oy)] for x in range(cols) for y in range(rows)]


def _make_seed_lines():
    """Seed: 12 points arranged on structured lines sharing intersections."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [1, 3], [2, 3], [3, 3], [4, 3],
        [0, 3], [1, 2], [2, 1],
        [0, 1]
    ]
    seen = set()
    clean = []
    for p in pts:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            clean.append(p[:])
    return clean


def _make_seed_two_lines():
    """Böröczky-type: 6 points on y=0, 6 points on y=1 with aligned diagonals."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0],
        [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
    ]
    return pts


def _make_seed_perturbed():
    """4x3 grid with one point moved to create extra 3-point lines."""
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [4, 3],
    ]
    return pts


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    best_pts = None
    best_score = float('inf')

    def _consider(pts):
        nonlocal best_pts, best_score
        if not _valid(pts):
            return
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Multiple seeds
    seeds = []
    seeds.append(_make_seed_grid(3, 4))
    seeds.append(_make_seed_grid(4, 3))
    seeds.append(_make_seed_lines())
    seeds.append(_make_seed_two_lines())
    seeds.append(_make_seed_perturbed())
    for scale in [2, 3]:
        seeds.append(_make_seed_grid(3, 4, 0, 0, scale))
        seeds.append(_make_seed_grid(4, 3, 0, 0, scale))

    # Evaluate all seeds
    for seed in seeds:
        if len(seed) == 12:
            _consider(seed)

    # Simulated annealing from best seed
    if best_pts is not None and time.time() < _DEADLINE - 20:
        pts_sa, score_sa = _simulated_annealing(best_pts, max_iters=250000, seed=7)
        if score_sa < best_score:
            best_score = score_sa
            best_pts = pts_sa

    # Local search from best so far
    if best_pts is not None and time.time() < _DEADLINE - 15:
        pts_ls, score_ls = _local_search(best_pts, max_iters=150000, seed=123)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # Another SA round with different seed
    if best_pts is not None and time.time() < _DEADLINE - 15:
        pts_sa2, score_sa2 = _simulated_annealing(best_pts, max_iters=200000, seed=42)
        if score_sa2 < best_score:
            best_score = score_sa2
            best_pts = pts_sa2

    # Local search again
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_ls2, score_ls2 = _local_search(best_pts, max_iters=100000, seed=99)
        if score_ls2 < best_score:
            best_score = score_ls2
            best_pts = pts_ls2

    # Deterministic polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts_polish, score_polish = _deterministic_polish(best_pts)
        if score_polish < best_score:
            best_score = score_polish
            best_pts = pts_polish

    if best_pts is None:
        best_pts = _make_seed_grid(3, 4)

    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
