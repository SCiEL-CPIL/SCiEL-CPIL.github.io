"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points.
    Returns (num_four, num_five_plus) where num_five_plus counts lines with >=5 points."""
    n = len(points)
    if n < 4:
        return 0, 0
    # For each pair of points, compute the line (normalized) and count how many points lie on it.
    # A line is identified by (A, B, C) where Ax + By + C = 0 with gcd(A,B,C)=1 and sign convention.
    line_counts = defaultdict(int)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            # Line through (xi,yi) and (xj,yj)
            # A(x - xi) + B(y - yi) = 0 where A = yj - yi, B = xi - xj
            A = yj - yi
            B = xi - xj
            # C = -(A*xi + B*yi)
            C = -(A * xi + B * yi)
            # Normalize: divide by gcd of A, B, C
            import math
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0 or C != 0) else 1
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            # Sign convention: first non-zero among A, B, C should be positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_counts[key] += 1
    # But each line with k points is counted C(k,2) times in line_counts.
    # We need to find for each distinct line, how many points are on it.
    # Better approach: for each line key, we need the actual count of points on that line.
    # Let's re-do: for each pair, we get a line. But multiple pairs on the same line give the same key.
    # So line_counts[key] = number of pairs on that line = C(k, 2) where k is the number of points on the line.
    # From C(k,2) we can recover k: k = (1 + sqrt(1 + 8*line_counts[key])) / 2
    
    num_four = 0
    num_five_plus = 0
    for key, pair_count in line_counts.items():
        # Solve C(k,2) = pair_count => k*(k-1)/2 = pair_count => k^2 - k - 2*pair_count = 0
        # k = (1 + sqrt(1 + 8*pair_count)) / 2
        import math
        disc = 1 + 8 * pair_count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc != disc:
            # Shouldn't happen for valid integer counts, but skip if not perfect square
            continue
        k = (1 + sqrt_disc) // 2
        if k == 4:
            num_four += 1
        elif k >= 5:
            num_five_plus += 1
    return num_four, num_five_plus

def _score(points):
    """Score = number of 4-point lines, penalized heavily if any 5-point line exists."""
    n4, n5 = _count_four_lines(points)
    if n5 > 0:
        return -1000000 + n4
    return n4

def _valid(points):
    """Check if points are valid: 16 distinct integer points, |x|,|y| <= 1000, no 5 collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    n4, n5 = _count_four_lines(points)
    return n5 == 0

def _local_search(points, max_iters=5000):
    """Local search: try moving individual points to nearby integer coordinates."""
    best = list(points)
    best_score = _score(best)
    n = len(best)
    
    for iteration in range(max_iters):
        if _remaining() < 5.0:
            break
        improved = False
        # Try moving each point to a nearby location
        for i in range(n):
            if _remaining() < 5.0:
                break
            xi, yi = best[i]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = xi + dx, yi + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check if this new position is already occupied
                    if [nx, ny] in best:
                        continue
                    new_points = list(best)
                    new_points[i] = [nx, ny]
                    s = _score(new_points)
                    if s > best_score:
                        best = new_points
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def _line_aligned_move(points, max_iters=3000):
    """Try moves that align points on existing or potential lines."""
    best = list(points)
    best_score = _score(best)
    n = len(best)
    
    for iteration in range(max_iters):
        if _remaining() < 5.0:
            break
        improved = False
        # For each pair of points, consider the line they define.
        # Try moving a third point onto that line to potentially create a 4-point line.
        for i in range(n):
            if _remaining() < 5.0:
                break
            for j in range(i + 1, n):
                if _remaining() < 5.0:
                    break
                xi, yi = best[i]
                xj, yj = best[j]
                # Direction vector
                dx = xj - xi
                dy = yj - yi
                # Try extending the line by small steps
                for step in [-2, -1, 1, 2]:
                    nx = xi + step * dx
                    ny = yi + step * dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    if [nx, ny] in best:
                        continue
                    # Find a point to move (not i or j)
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        old_x, old_y = best[k]
                        new_points = list(best)
                        new_points[k] = [nx, ny]
                        s = _score(new_points)
                        if s > best_score:
                            best = new_points
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_score

def _random_start():
    """Generate a random starting configuration."""
    points = []
    used = set()
    while len(points) < 16:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in used:
            used.add((x, y))
            points.append([x, y])
    return points

def _grid_start():
    """4x4 grid starting configuration."""
    return [[x, y] for x in range(4) for y in range(4)]

def _perturbed_grid():
    """4x4 grid with small random perturbations."""
    base = _grid_start()
    points = []
    used = set()
    for x, y in base:
        # Try to place near original position
        placed = False
        for _ in range(20):
            nx = x + random.randint(-2, 2)
            ny = y + random.randint(-2, 2)
            if (nx, ny) not in used:
                used.add((nx, ny))
                points.append([nx, ny])
                placed = True
                break
        if not placed:
            # Find an empty spot
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in used:
                        used.add((nx, ny))
                        points.append([nx, ny])
                        break
                else:
                    continue
                break
    return points

def construct():
    best_points = _grid_start()
    best_score = _score(best_points)
    
    # Try multiple strategies
    strategies = [
        ("grid", _grid_start()),
        ("perturbed_grid", _perturbed_grid()),
        ("random1", _random_start()),
        ("random2", _random_start()),
        ("random3", _random_start()),
    ]
    
    for name, start in strategies:
        if _remaining() < 10.0:
            break
        
        # Local search
        pts, sc = _local_search(start, max_iters=3000)
        if sc > best_score:
            best_points = pts
            best_score = sc
        
        if _remaining() < 10.0:
            break
        
        # Line-aligned search
        pts, sc = _line_aligned_move(pts, max_iters=2000)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    # Final verification
    if _valid(best_points):
        return best_points
    else:
        # Fallback to known valid configuration
        return _grid_start()
# EVOLVE-BLOCK-END
