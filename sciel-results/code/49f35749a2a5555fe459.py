"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations
from collections import defaultdict
import math
import random

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Normalize: make dx positive, or if dx==0 make dy positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Line: dy*x - dx*y = c where c = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _score(points):
    """Count number of lines containing exactly 4 of the points.
    Returns -1 if invalid (5+ collinear)."""
    n = len(points)
    if n != 16:
        return -1
    # Check bounds and distinctness
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        if (x, y) in seen:
            return -1
        seen.add((x, y))
    
    # Build line map
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_counts:
                line_counts[key] = 2
            else:
                line_counts[key] += 1
    
    # For each line, count actual points on it
    count4 = 0
    for key, cnt in line_counts.items():
        if cnt < 4:
            continue
        # Count actual points on this line
        dx, dy, c = key
        actual = 0
        for x, y in points:
            if dy * x - dx * y == c:
                actual += 1
        if actual >= 5:
            return -1
        if actual == 4:
            count4 += 1
    
    return count4

def _is_valid(points):
    """Check if points are valid: 16 distinct, |x|,|y|<=1000, no 5 collinear."""
    return _score(points) >= 0

def _local_search(points, deadline, rng):
    """Improve points via local search with multiple strategies."""
    best = [list(p) for p in points]
    best_score = _score(best)
    if best_score < 0:
        best_score = 0
    
    while time.time() < deadline:
        improved = False
        
        # Strategy 1: Single point moves with small offsets
        for idx in range(16):
            if time.time() > deadline:
                break
            x, y = best[idx]
            for dx in range(-5, 6):
                if time.time() > deadline:
                    break
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_points = [list(p) for p in best]
                    new_points[idx] = [nx, ny]
                    if len(set(tuple(p) for p in new_points)) != 16:
                        continue
                    s = _score(new_points)
                    if s > best_score:
                        best = new_points
                        best_score = s
                        improved = True
                        x, y = nx, ny
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Strategy 2: Larger random single-point moves
        found_improvement = False
        for _ in range(100):
            if time.time() > deadline:
                break
            idx = rng.randint(0, 15)
            x, y = best[idx]
            nx = x + rng.randint(-20, 20)
            ny = y + rng.randint(-20, 20)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            new_points = [list(p) for p in best]
            new_points[idx] = [nx, ny]
            if len(set(tuple(p) for p in new_points)) != 16:
                continue
            s = _score(new_points)
            if s > best_score:
                best = new_points
                best_score = s
                found_improvement = True
                break
        
        if found_improvement:
            continue
        
        # Strategy 3: Simulated annealing style - accept some worse moves
        accepted_worse = False
        for _ in range(50):
            if time.time() > deadline:
                break
            idx = rng.randint(0, 15)
            x, y = best[idx]
            nx = x + rng.randint(-10, 10)
            ny = y + rng.randint(-10, 10)
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
            new_points = [list(p) for p in best]
            new_points[idx] = [nx, ny]
            if len(set(tuple(p) for p in new_points)) != 16:
                continue
            s = _score(new_points)
            if s < 0:
                continue
            # Accept worse moves with small probability to escape local optima
            if s >= best_score or rng.random() < 0.1:
                best = new_points
                if s > best_score:
                    best_score = s
                accepted_worse = True
                break
        
        if accepted_worse:
            continue
        
        # Strategy 4: Random restart with perturbation
        restart = [list(p) for p in best]
        for _ in range(3):
            idx = rng.randint(0, 15)
            restart[idx] = [best[idx][0] + rng.randint(-15, 15), 
                           best[idx][1] + rng.randint(-15, 15)]
        if len(set(tuple(p) for p in restart)) == 16:
            s = _score(restart)
            if s >= 0:
                best = restart
                best_score = s
        
        # Check if we've been stuck
        if time.time() > deadline:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    # Initial configurations to try
    initial_configs = [
        # 4x4 grid
        [[x, y] for x in range(4) for y in range(4)],
        # 4x4 grid shifted
        [[x + 1, y + 1] for x in range(4) for y in range(4)],
        # 2x8 grid
        [[x, y] for x in range(2) for y in range(8)],
        # 8x2 grid
        [[x, y] for x in range(8) for y in range(2)],
        # Perturbed 4x4 grid
        [[x, y] for x in range(4) for y in range(4) if (x, y) != (1, 1)] + [[5, 5]],
    ]
    
    best_points = None
    best_score = -1
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        s = _score(config)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in config]
    
    if best_points is None or best_score < 0:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = _score(best_points)
    
    # Run local search from best initial config
    best_points, best_score = _local_search(best_points, deadline, rng)
    
    # Try more random restarts with different seeds
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        for seed in range(5):
            if time.time() > deadline:
                break
            rng2 = random.Random(seed * 1000 + 7)
            # Create a random starting configuration
            random_points = []
            attempts = 0
            while len(random_points) < 16 and attempts < 200:
                x = rng2.randint(-50, 50)
                y = rng2.randint(-50, 50)
                if (x, y) not in [(p[0], p[1]) for p in random_points]:
                    random_points.append([x, y])
                attempts += 1
            
            if len(random_points) == 16:
                s = _score(random_points)
                if s >= 0:
                    # Run local search from this random start
                    rp, rs = _local_search(random_points, deadline, rng2)
                    if rs > best_score:
                        best_score = rs
                        best_points = rp
    
    # Final verification
    final_score = _score(best_points)
    if final_score < 0:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
