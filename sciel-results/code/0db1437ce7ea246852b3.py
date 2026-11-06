"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt

def _line_key(p1, p2):
    """Return a canonical key for the line through two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Line equation: dx*x + dy*y + c = 0
    c = -(dx * y1 + dy * x1)
    # Normalize so that gcd(dx, dy, c) = 1 and leading coefficient positive
    g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if g2 > 0:
        dx //= g2
        dy //= g2
        c //= g2
    if dx < 0 or (dx == 0 and dy < 0) or (dx == 0 and dy == 0 and c < 0):
        dx, dy, c = -dx, -dy, -c
    return (dx, dy, c)

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    # Map each line to a canonical tuple; count pairs per line
    line_pair_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
    
    ordinary = 0
    for key, m in line_pair_counts.items():
        # m = C(k,2) where k = number of points on line
        # k^2 - k - 2m = 0 => k = (1 + sqrt(1+8m)) / 2
        disc = 1 + 8 * m
        sqrt_d = isqrt(disc)
        if sqrt_d * sqrt_d == disc:
            k = (1 + sqrt_d) // 2
            if k == 2:
                ordinary += 1
        else:
            # Should not happen for valid integer point sets
            ordinary += 1
    return ordinary

def _eval_config(points):
    """Return (num_ordinary, valid) for a configuration of 12 points."""
    if len(points) != 12:
        return (999, False)
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 12:
        return (999, False)
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return (999, False)
    # Check not all collinear
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    all_col = True
    for x, y in points[2:]:
        if dx * (y - y0) != dy * (x - x0):
            all_col = False
            break
    if all_col:
        return (999, False)
    return (_count_ordinary(points), True)

def _random_points():
    """Generate 12 random integer points within bounds."""
    pts = []
    while len(pts) < 12:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if [x, y] not in pts:
            pts.append([x, y])
    return pts

def _local_search(points, deadline, max_iter=2000, radius=10):
    """Local search: try moving individual points to reduce ordinary lines."""
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try moving each point to a nearby location
        for i in range(12):
            if time.time() > deadline:
                break
            x, y = current[i]
            # Try a set of candidate moves
            candidates = []
            for dx in range(-radius, radius + 1, 2):
                for dy in range(-radius, radius + 1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                        candidates.append((new_x, new_y))
            
            # Also try some random larger moves
            for _ in range(5):
                new_x = x + random.randint(-30, 30)
                new_y = y + random.randint(-30, 30)
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    candidates.append((new_x, new_y))
            
            # Shuffle candidates for diversity
            random.shuffle(candidates)
            
            for new_x, new_y in candidates:
                if time.time() > deadline:
                    break
                trial = [p[:] for p in current]
                trial[i] = [new_x, new_y]
                # Check distinctness
                if len(set(tuple(p) for p in trial)) != 12:
                    continue
                score, valid = _eval_config(trial)
                if valid and score < current_score:
                    current_score = score
                    current = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try a random restart within local neighborhood
            if random.random() < 0.1:
                i = random.randint(0, 11)
                x, y = current[i]
                new_x = x + random.randint(-20, 20)
                new_y = y + random.randint(-20, 20)
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    trial = [p[:] for p in current]
                    trial[i] = [new_x, new_y]
                    if len(set(tuple(p) for p in trial)) == 12:
                        score, valid = _eval_config(trial)
                        if valid and score <= current_score:
                            current = trial
                            current_score = score
    return current, current_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    random.seed(42)  # Deterministic for reproducibility
    
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary(best_points)
    
    # Try several structured configurations as starting points
    candidates = []
    
    # 1. 3x4 grid (baseline)
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # 2. 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # 3. Symmetric cross pattern
    # Points on axes and diagonals
    cross_pts = [
        [0, 0],
        [1, 0], [-1, 0], [2, 0], [-2, 0],
        [0, 1], [0, -1], [0, 2], [0, -2],
        [1, 1], [1, -1], [-1, 1]
    ]
    candidates.append(cross_pts)
    
    # 4. Hexagonal arrangement: center + 6 around + 5 more
    hex_pts = [
        [0, 0],
        [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1],
        [2, 0], [-2, 0], [0, 2], [0, -2], [2, 2]
    ]
    candidates.append(hex_pts)
    
    # 5. Try a configuration with more collinearities
    # 4 points on y=0, 4 on y=1, 4 on y=2 but shifted
    shifted = []
    for y in range(3):
        for x in range(4):
            shifted.append([x + y, y])
    candidates.append(shifted)
    
    # 6. Another structured pattern
    pattern6 = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [4, 2]
    ]
    candidates.append(pattern6)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score, valid = _eval_config(cand)
        if valid and score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    # Local search from best candidate
    if time.time() < deadline - 10:
        best_points, best_score = _local_search(best_points, deadline, max_iter=3000, radius=8)
    
    # Try random restarts if time permits
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline:
                break
            random_pts = _random_points()
            score, valid = _eval_config(random_pts)
            if valid and score < best_score:
                best_score = score
                best_points = [p[:] for p in random_pts]
                # Local search from this better starting point
                if time.time() < deadline - 10:
                    best_points, best_score = _local_search(best_points, deadline, max_iter=1000, radius=8)
    
    # Final local search polish
    if time.time() < deadline - 2:
        best_points, best_score = _local_search(best_points, deadline, max_iter=1000, radius=5)
    
    return best_points
# EVOLVE-BLOCK-END
