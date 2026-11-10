"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 0, x1)  # vertical line x = x1
    if dy == 0:
        return (0, 1, y1)  # horizontal line y = y1
    # General line: ax + by = c
    # Normalize so that a >= 0, and if a == 0 then b > 0
    a = dy
    b = -dx
    c = a * x1 + b * y1
    # Normalize
    g = gcd(abs(a), abs(b))
    if g > 1:
        a //= g
        b //= g
        c //= g
    if a < 0:
        a = -a
        b = -b
        c = -c
    elif a == 0 and b < 0:
        b = -b
        c = -c
    return (a, b, c)

def _count_4_points(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    # Map from line key to set of point indices on that line
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    count = 0
    for key, idx_set in line_map.items():
        if len(idx_set) == 4:
            count += 1
    return count

def _is_valid(points):
    """Check that we have 16 distinct points with |x|,|y| <= 1000 and no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check no 5 collinear
    n = 16
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_map[key].add(i)
                line_map[key].add(j)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 5:
            return False
    return True

def _local_search(points, max_time=100):
    """Local search to maximize 4-point lines."""
    start_time = time.time()
    current = [list(p) for p in points]
    current_score = _count_4_points(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    # Neighborhood: move one point by small offset
    offsets = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx != 0 or dy != 0:
                offsets.append((dx, dy))
    random.shuffle(offsets)
    
    # Also try larger moves occasionally
    for _ in range(20):
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        if dx != 0 or dy != 0:
            offsets.append((dx, dy))
    
    while time.time() - start_time < max_time:
        # Pick a random point and offset
        idx = random.randint(0, 15)
        dx, dy = random.choice(offsets)
        new_points = [list(p) for p in current]
        new_x = new_points[idx][0] + dx
        new_y = new_points[idx][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        new_points[idx] = [new_x, new_y]
        if len(set(tuple(p) for p in new_points)) != 16:
            continue
        new_score = _count_4_points(new_points)
        # Check no 5 collinear
        n = 16
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = _line_key(new_points[i], new_points[j])
                if key is not None:
                    line_map[key].add(i)
                    line_map[key].add(j)
        valid = True
        for key, idx_set in line_map.items():
            if len(idx_set) >= 5:
                valid = False
                break
        if not valid:
            continue
        if new_score >= current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            # Occasional acceptance for exploration
            if random.random() < 0.1:
                current = new_points
                current_score = new_score
    return best, best_score

def _generate_initial():
    """Generate initial 4x4 grid centered at origin."""
    return [[x, y] for x in range(-1, 3) for y in range(-1, 3)]

def construct():
    start_time = time.time()
    deadline = 110  # Keep 40s safety margin
    
    # Start with 4x4 grid
    initial = _generate_initial()
    best_points, best_score = _local_search(initial, max_time=min(50, deadline - (time.time() - start_time)))
    
    # Try a few random restarts if time permits
    for _ in range(5):
        if time.time() - start_time > deadline:
            break
        # Perturb initial grid
        perturbed = []
        for x, y in initial:
            if random.random() < 0.3:
                nx = x + random.randint(-2, 2)
                ny = y + random.randint(-2, 2)
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    perturbed.append([nx, ny])
                else:
                    perturbed.append([x, y])
            else:
                perturbed.append([x, y])
        if len(set(tuple(p) for p in perturbed)) == 16:
            pts, score = _local_search(perturbed, max_time=min(30, deadline - (time.time() - start_time)))
            if score > best_score:
                best_points, best_score = pts, score
    
    # Final validation
    if not _is_valid(best_points):
        return initial
    return best_points
# EVOLVE-BLOCK-END
