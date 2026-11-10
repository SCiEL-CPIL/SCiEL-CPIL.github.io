import time
import random
import numpy as np
from collections import Counter

def _doubled_areas(points):
    """Compute absolute doubled areas for all triples."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(pts[i,0]*(pts[j,1]-pts[k,1]) + pts[j,0]*(pts[k,1]-pts[i,1]) + pts[k,0]*(pts[i,1]-pts[j,1]))
                if det > 0:
                    areas.append(det)
    return areas

def _score(points):
    """Return the maximum multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _valid(points):
    """Check that points are exactly 30 distinct integer pairs in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set()
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _neighborhood_move(points, rng):
    """Generate a single neighbor by moving one point to a random valid position."""
    new_points = list(points)
    idx = rng.randrange(30)
    # Try to find a new position
    for _ in range(100):
        nx = rng.randint(-30, 30)
        ny = rng.randint(-30, 30)
        if (nx, ny) not in set(new_points) or (nx, ny) == new_points[idx]:
            new_points[idx] = (nx, ny)
            break
    return new_points

def _try_improve(points, score, rng, time_limit):
    """Local search: hill-climbing with random restarts within time limit."""
    best_points = list(points)
    best_score = score
    start = time.time()
    
    current_points = list(points)
    current_score = score
    
    while time.time() - start < time_limit:
        # Try several random moves
        improved = False
        for _ in range(50):
            if time.time() - start >= time_limit:
                break
            neighbor = _neighborhood_move(current_points, rng)
            if neighbor == current_points:
                continue
            n_score = _score(neighbor)
            if n_score > current_score:
                current_points = neighbor
                current_score = n_score
                improved = True
                if n_score > best_score:
                    best_score = n_score
                    best_points = list(neighbor)
                break
        if not improved:
            # Stuck: try a random restart from a perturbed version
            perturbed = list(current_points)
            for _ in range(3):
                idx = rng.randrange(30)
                for _ in range(50):
                    nx = rng.randint(-30, 30)
                    ny = rng.randint(-30, 30)
                    if (nx, ny) not in set(perturbed):
                        perturbed[idx] = (nx, ny)
                        break
            current_points = perturbed
            current_score = _score(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = list(current_points)
    
    return best_points, best_score

def construct():
    deadline = 90.0
    start = time.time()
    rng = random.Random(42)
    
    # Initial configuration: try several structured seeds
    candidates = []
    
    # Seed 1: 6x5 grid (original)
    candidates.append([(x, y) for x in range(-2, 4) for y in range(-2, 3)])
    
    # Seed 2: 5x6 grid
    candidates.append([(x, y) for x in range(-2, 3) for y in range(-3, 3)])
    
    # Seed 3: 10x3 grid
    candidates.append([(x, y) for x in range(-5, 5) for y in range(-1, 2)])
    
    # Seed 4: Random points
    pts = set()
    while len(pts) < 30:
        pts.add((rng.randint(-30, 30), rng.randint(-30, 30)))
    candidates.append(list(pts))
    
    # Seed 5: Clustered points
    pts = set()
    for cx, cy in [(-10, -10), (0, 0), (10, 10)]:
        for _ in range(10):
            for _ in range(100):
                x = cx + rng.randint(-4, 4)
                y = cy + rng.randint(-4, 4)
                if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                    pts.add((x, y))
                    break
    if len(pts) >= 30:
        candidates.append(list(pts)[:30])
    
    # Evaluate all seeds
    best_points = None
    best_score = 0
    for cand in candidates:
        if _valid(cand):
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = list(cand)
    
    if best_points is None:
        best_points = candidates[0]
        best_score = _score(best_points)
    
    # Local search with time budget
    remaining = deadline - (time.time() - start) - 5.0  # 5s safety margin
    if remaining > 1.0:
        best_points, best_score = _try_improve(best_points, best_score, rng, remaining)
    
    # Final validation
    if not _valid(best_points):
        best_points = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    
    return best_points
