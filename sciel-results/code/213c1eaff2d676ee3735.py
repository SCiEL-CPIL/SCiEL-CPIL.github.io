"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from math import gcd

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    line_count = defaultdict(int)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            line_key = (dx, dy, c)
            line_count[line_key] += 1
    
    # For each line, count how many points are on it
    # A line defined by (dx, dy, c) has k points if there are k points satisfying dy*x - dx*y = c
    # But we counted pairs, so if k points are on the line, we counted C(k,2) pairs
    # We need to find k for each line
    count = 0
    for line_key, pair_count in line_count.items():
        # pair_count = C(k,2) = k*(k-1)/2
        # Solve for k
        # k*(k-1)/2 = pair_count
        # k^2 - k - 2*pair_count = 0
        # k = (1 + sqrt(1 + 8*pair_count)) / 2
        import math
        disc = 1 + 8 * pair_count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc == disc and (1 + sqrt_disc) % 2 == 0:
            k = (1 + sqrt_disc) // 2
            if k == 4:
                count += 1
    
    return count

def _no_five_collinear(points):
    """Check that no line contains 5 or more points."""
    n = len(points)
    line_count = defaultdict(int)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            line_key = (dx, dy, c)
            line_count[line_key] += 1
    
    for line_key, pair_count in line_count.items():
        # pair_count = C(k,2)
        import math
        disc = 1 + 8 * pair_count
        sqrt_disc = int(math.isqrt(disc))
        if sqrt_disc * sqrt_disc == disc and (1 + sqrt_disc) % 2 == 0:
            k = (1 + sqrt_disc) // 2
            if k >= 5:
                return False
    return True

def _valid(points):
    """Check all constraints."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if not _no_five_collinear(points):
        return False
    return True

def _score(points):
    """Score = number of lines with exactly 4 points, 0 if invalid."""
    if not _valid(points):
        return 0
    return _count_lines_exactly_4(points)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    random.seed(42)
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    # Hill climbing with random restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Different initial configurations
        if restart == 0:
            current = [[x, y] for x in range(4) for y in range(4)]
        elif restart == 1:
            # Slightly perturbed grid
            current = [[x, y] for x in range(4) for y in range(4)]
            for _ in range(3):
                idx = random.randint(0, 15)
                current[idx] = [random.randint(-5, 5), random.randint(-5, 5)]
        elif restart == 2:
            # Another perturbation
            current = [[x, y] for x in range(4) for y in range(4)]
            for _ in range(5):
                idx = random.randint(0, 15)
                current[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        else:
            # Random configuration
            current = []
            for _ in range(16):
                current.append([random.randint(-20, 20), random.randint(-20, 20)])
        
        if not _valid(current):
            continue
        
        cur_score = _score(current)
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try moving each point
            for i in range(16):
                if time.time() > deadline:
                    break
                old_x, old_y = current[i]
                # Try nearby positions
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if abs(new_x) > 1000 or abs(new_y) > 1000:
                            continue
                        candidate = [p[:] for p in current]
                        candidate[i] = [new_x, new_y]
                        cand_score = _score(candidate)
                        if cand_score > cur_score:
                            cur_score = cand_score
                            current = candidate
                            improved = True
                            if cur_score > best_score:
                                best_score = cur_score
                                best_points = [p[:] for p in current]
        
        # Also try larger random moves
        for _ in range(50):
            if time.time() > deadline:
                break
            i = random.randint(0, 15)
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            candidate = [p[:] for p in current]
            candidate[i] = [new_x, new_y]
            cand_score = _score(candidate)
            if cand_score > cur_score:
                cur_score = cand_score
                current = candidate
                if cur_score > best_score:
                    best_score = cur_score
                    best_points = [p[:] for p in current]
    
    return best_points
# EVOLVE-BLOCK-END
