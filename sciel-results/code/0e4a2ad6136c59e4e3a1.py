import random
import time
import math
from collections import defaultdict
from itertools import combinations

# Deadline guard: evaluator kills after 150s; we aim to finish well before.
_DEADLINE = 130.0  # seconds from start of construct()

def _line_key(p1, p2):
    """Return a canonical key for the line through points p1=(x1,y1) and p2=(x2,y2).
    The key is (dx, dy, c) where dx, dy are gcd-reduced and c is the offset,
    all normalized so that the representation is unique up to sign."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize sign: make the first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line equation: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_exactly_four_lines(points):
    """Count lines that contain exactly 4 of the given points.
    points: list of (x, y) tuples, len 16."""
    # Map each line key to the set of points on it
    line_points = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_points[key].add(i)
    count = 0
    for key, pts in line_points.items():
        if len(pts) == 4:
            count += 1
    return count

def _is_valid(points):
    """Check: 16 distinct integer points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check no five collinear
    line_points = defaultdict(set)
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_points[key].add(i)
    for key, pts in line_points.items():
        if len(pts) >= 5:
            return False
    return True

def _score(points):
    """Return the number of lines with exactly 4 points, or -1 if invalid."""
    if not _is_valid(points):
        return -1
    return _count_exactly_four_lines(points)

def _make_grid_4x4():
    """4x4 grid: 16 points at (0..3, 0..3)."""
    return [(x, y) for x in range(4) for y in range(4)]

def _make_perturbed_grid(seed_rng, scale=1):
    """Perturb a 4x4 grid slightly to break symmetries while keeping structure."""
    pts = []
    for x in range(4):
        for y in range(4):
            # Small random perturbation
            px = x * scale + seed_rng.randint(-2, 2)
            py = y * scale + seed_rng.randint(-2, 2)
            pts.append((px, py))
    # Ensure distinctness
    unique = []
    seen = set()
    for p in pts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    while len(unique) < 16:
        # Add random point
        px = seed_rng.randint(-1000, 1000)
        py = seed_rng.randint(-1000, 1000)
        if (px, py) not in seen:
            seen.add((px, py))
            unique.append((px, py))
    return unique[:16]

def _make_random_points(seed_rng, coord_range=200):
    """Generate 16 random distinct integer points."""
    pts = []
    seen = set()
    while len(pts) < 16:
        x = seed_rng.randint(-coord_range, coord_range)
        y = seed_rng.randint(-coord_range, coord_range)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
    return pts

def _local_search(points, seed_rng, max_steps=2000, time_limit=None):
    """Hill-climbing local search: move one point at a time to improve score."""
    current = list(points)
    current_score = _score(current)
    if current_score < 0:
        return current, current_score
    
    best = list(current)
    best_score = current_score
    
    start_time = time.time()
    step = 0
    while step < max_steps:
        if time_limit and (time.time() - start_time) > time_limit:
            break
        
        # Try moving each point
        improved = False
        for i in range(16):
            # Try several random new positions for point i
            for _ in range(5):
                new_x = current[i][0] + seed_rng.randint(-50, 50)
                new_y = current[i][1] + seed_rng.randint(-50, 50)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                candidate = list(current)
                candidate[i] = (new_x, new_y)
                # Check distinctness
                if len(set(candidate)) != 16:
                    continue
                s = _score(candidate)
                if s > current_score:
                    current = candidate
                    current_score = s
                    if s > best_score:
                        best = list(current)
                        best_score = s
                    improved = True
                    break  # Accept first improvement for this point
        if not improved:
            # Stuck: try a random kick
            i = seed_rng.randint(0, 15)
            new_x = seed_rng.randint(-1000, 1000)
            new_y = seed_rng.randint(-1000, 1000)
            candidate = list(current)
            candidate[i] = (new_x, new_y)
            if len(set(candidate)) == 16:
                s = _score(candidate)
                if s >= 0:  # Accept if valid, even if worse
                    current = candidate
                    current_score = s
        step += 1
    
    return best, best_score

def _simulated_annealing(points, seed_rng, start_temp=10.0, end_temp=0.01, 
                          max_steps=5000, time_limit=None):
    """Simulated annealing to escape local optima."""
    current = list(points)
    current_score = _score(current)
    if current_score < 0:
        return current, current_score
    
    best = list(current)
    best_score = current_score
    
    start_time = time.time()
    step = 0
    while step < max_steps:
        if time_limit and (time.time() - start_time) > time_limit:
            break
        
        # Temperature schedule
        progress = step / max_steps
        temp = start_temp * math.exp(-10 * progress)
        
        # Generate a neighbor: move one random point
        i = seed_rng.randint(0, 15)
        new_x = current[i][0] + seed_rng.randint(-30, 30)
        new_y = current[i][1] + seed_rng.randint(-30, 30)
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            step += 1
            continue
        candidate = list(current)
        candidate[i] = (new_x, new_y)
        if len(set(candidate)) != 16:
            step += 1
            continue
        
        s = _score(candidate)
        if s < 0:
            step += 1
            continue
        
        delta = s - current_score
        if delta >= 0:
            current = candidate
            current_score = s
            if s > best_score:
                best = list(current)
                best_score = s
        else:
            # Accept with probability exp(delta/temp)
            if temp > 0 and seed_rng.random() < math.exp(delta / temp):
                current = candidate
                current_score = s
        
        step += 1
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    seed_rng = random.Random(42)
    
    # Initial candidates
    candidates = []
    
    # 1. 4x4 grid
    candidates.append(_make_grid_4x4())
    
    # 2. Perturbed grids with different scales
    for scale in [1, 2, 3, 5]:
        candidates.append(_make_perturbed_grid(seed_rng, scale=scale))
    
    # 3. Random points
    for _ in range(5):
        candidates.append(_make_random_points(seed_rng, coord_range=100))
    
    # 4. Structured patterns: two parallel 4x4 grids shifted
    for dx in [4, 5, 7]:
        pts = []
        for x in range(4):
            for y in range(4):
                pts.append((x * dx, y * dx))
        candidates.append(pts)
    
    # Evaluate all candidates
    best_points = None
    best_score = -1
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
    
    # Local search on best candidate
    if best_points is not None:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            ls_points, ls_score = _local_search(
                best_points, seed_rng, max_steps=3000, time_limit=min(remaining * 0.4, 30)
            )
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_points
    
    # Simulated annealing from multiple starts
    if best_points is not None:
        remaining = deadline - time.time() - 5
        if remaining > 20:
            # SA from best
            sa_points, sa_score = _simulated_annealing(
                best_points, seed_rng, start_temp=5.0, max_steps=4000,
                time_limit=min(remaining * 0.3, 25)
            )
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points
            
            # SA from a perturbed grid
            remaining = deadline - time.time() - 5
            if remaining > 15:
                pert = _make_perturbed_grid(seed_rng, scale=2)
                sa_points2, sa_score2 = _simulated_annealing(
                    pert, seed_rng, start_temp=8.0, max_steps=3000,
                    time_limit=min(remaining * 0.5, 20)
                )
                if sa_score2 > best_score:
                    best_score = sa_score2
                    best_points = sa_points2
    
    # Final local search polish
    if best_points is not None:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            final_points, final_score = _local_search(
                best_points, seed_rng, max_steps=2000, time_limit=min(remaining, 10)
            )
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
    
    # Ensure we have a valid result
    if best_points is None or _score(best_points) < 0:
        # Fallback: 4x4 grid
        best_points = _make_grid_4x4()
    
    # Convert to list of [x, y] as required
    return [[x, y] for x, y in best_points]
