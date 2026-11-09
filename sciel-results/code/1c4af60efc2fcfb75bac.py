import time
import random
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return a canonical key for the line through two points (dx, dy, c)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    # Normalize direction
    from math import gcd
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Ensure canonical sign: first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = c
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _count_exact4(points):
    """Count lines containing exactly 4 of the given points. Also return dict of line->count for reuse."""
    line_counts = defaultdict(int)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_counts[key] += 1
    exact4 = 0
    for key, cnt in line_counts.items():
        if cnt == 4:
            exact4 += 1
    return exact4, line_counts

def _is_valid(points):
    """Check no five collinear."""
    _, line_counts = _count_exact4(points)
    for cnt in line_counts.values():
        if cnt >= 5:
            return False
    return True

def _score(points):
    """Return (exact4_count, -max_collinear) as objective; higher is better."""
    exact4, line_counts = _count_exact4(points)
    max_c = max(line_counts.values()) if line_counts else 0
    # Penalize any 5+ collinear heavily
    if max_c >= 5:
        return -1000000 + exact4, -max_c
    return exact4, -max_c

def _neighborhood_move(points, rng):
    """Generate a random neighbor by moving one point slightly."""
    new_points = [p[:] for p in points]
    idx = rng.randint(0, len(new_points) - 1)
    ox, oy = new_points[idx]
    # Try a random offset
    for _ in range(50):
        nx = ox + rng.randint(-3, 3)
        ny = oy + rng.randint(-3, 3)
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            # Check distinctness
            dup = False
            for i, p in enumerate(new_points):
                if i != idx and p[0] == nx and p[1] == ny:
                    dup = True
                    break
            if not dup:
                new_points[idx] = [nx, ny]
                return new_points
    return None

def _local_search(points, rng, time_limit):
    """Hill-climbing with random restarts within time limit."""
    best = [p[:] for p in points]
    best_score = _score(best)
    start = time.time()
    
    cur = [p[:] for p in points]
    cur_score = _score(cur)
    
    stuck = 0
    while time.time() - start < time_limit:
        if time.time() - start > time_limit:
            break
        neighbor = _neighborhood_move(cur, rng)
        if neighbor is None:
            stuck += 1
            if stuck > 100:
                # Restart from best
                cur = [p[:] for p in best]
                cur_score = best_score
                stuck = 0
            continue
        stuck = 0
        n_score = _score(neighbor)
        if n_score > cur_score:
            cur = neighbor
            cur_score = n_score
            if cur_score > best_score:
                best = [p[:] for p in cur]
                best_score = cur_score
        # Small probability of accepting worse move (simulated annealing lite)
        elif rng.random() < 0.05:
            cur = neighbor
            cur_score = n_score
    
    return best, best_score

def _seed_grid():
    """4x4 grid seed."""
    return [[x, y] for x in range(4) for y in range(4)]

def _seed_diagonal():
    """A seed with more diagonal structure."""
    pts = []
    # 4x4 grid
    for x in range(4):
        for y in range(4):
            pts.append([x, y])
    return pts

def _seed_random():
    """Random seed in a small box."""
    rng = random.Random(42)
    pts = set()
    while len(pts) < 16:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        pts.add((x, y))
    return [list(p) for p in pts]

def construct():
    deadline = time.time() + 140  # 140s to be safe within 150s
    rng = random.Random(12345)
    
    # Multiple seed strategies
    seeds = [
        _seed_grid(),
        _seed_random(),
    ]
    
    # Add some structured seeds
    # Grid with shifted rows
    seed3 = []
    for i in range(4):
        for j in range(4):
            seed3.append([i + j, j])  # sheared grid
    seeds.append(seed3)
    
    # Another structured seed: two 4x2 blocks
    seed4 = []
    for i in range(4):
        for j in range(2):
            seed4.append([i, j])
            seed4.append([i + 4, j + 3])
    seeds.append(seed4)
    
    best_points = None
    best_score = (-1, -1)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        # Verify seed is valid
        if not _is_valid(seed):
            continue
        s_score = _score(seed)
        if s_score > best_score:
            best_score = s_score
            best_points = [p[:] for p in seed]
    
    if best_points is None:
        best_points = _seed_grid()
        best_score = _score(best_points)
    
    # Local search on best seed
    remaining = deadline - time.time()
    if remaining > 5:
        improved, imp_score = _local_search(best_points, rng, remaining - 5)
        if imp_score > best_score:
            best_points = improved
            best_score = imp_score
    
    # Additional local search from different seeds if time permits
    remaining = deadline - time.time()
    if remaining > 10:
        for seed in seeds:
            if time.time() > deadline:
                break
            if not _is_valid(seed):
                continue
            s_score = _score(seed)
            if s_score >= best_score[0] - 2:  # Only if competitive
                improved, imp_score = _local_search(seed, rng, remaining - 5)
                if imp_score > best_score:
                    best_points = improved
                    best_score = imp_score
                remaining = deadline - time.time()
                if remaining < 10:
                    break
    
    # Final verification
    if not _is_valid(best_points):
        # Fallback to grid
        best_points = _seed_grid()
    
    return best_points

# EVOLVE-BLOCK-END
