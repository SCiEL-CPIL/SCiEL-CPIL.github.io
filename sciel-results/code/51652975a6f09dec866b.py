<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # A 4 x 4 grid: rows, columns and the two main diagonals.
    return [[x, y] for x in range(4) for y in range(4)]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _line_key(p1, p2):
    """Return canonical key for the line through two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return ('v', p1[0])
    if dy == 0:
        return ('h', p1[1])
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Canonical form: line through p1 with direction (dx,dy)
    # Use cross product for offset: dy*x - dx*y = const
    c = dy * p1[0] - dx * p1[1]
    return ('d', dx, dy, c)

def _count_4point_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # For each line, count how many points lie on it
    # A line with C pairs has C = k*(k-1)/2 where k is the number of points
    four_count = 0
    for key, pair_count in line_counts.items():
        # Find k such that k*(k-1)/2 = pair_count
        # k = (1 + sqrt(1 + 8*pair_count)) / 2
        disc = 1 + 8 * pair_count
        k_sq = int(disc ** 0.5)
        if k_sq * k_sq == disc:
            k = (1 + k_sq) // 2
            if k == 4:
                four_count += 1
    return four_count

def _validate(points):
    """Check that points are valid: 16 distinct integer points, |x|,|y| <= 1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    
    # Check no 5 collinear
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            count = 1
            for k in range(n):
                if k != i and k != j:
                    if _line_key(points[i], points[k]) == key:
                        count += 1
            if count >= 5:
                return False
    return True

def _generate_initial_5x5():
    """Generate a 5x5 grid subset (16 points from 25)."""
    full_grid = [(x, y) for x in range(5) for y in range(5)]
    random.shuffle(full_grid)
    return [list(p) for p in full_grid[:16]]

def _generate_initial_6x5():
    """Generate a 6x5 grid subset (16 points from 30)."""
    full_grid = [(x, y) for x in range(6) for y in range(5]
    random.shuffle(full_grid)
    return [list(p) for p in full_grid[:16]]

def _generate_initial_4x4_scaled():
    """Generate a scaled 4x4 grid."""
    scale = random.choice([1, 2, 3, 4])
    return [[x * scale, y * scale] for x in range(4) for y in range(4)]

def _local_search(points, max_iter=5000, deadline=None):
    """Perform local search to maximize 4-point lines."""
    current = [p[:] for p in points]
    current_score = _count_4point_lines(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point and try moving it
        idx = random.randint(0, 15)
        old_point = current[idx][:]
        
        # Generate candidate moves: nearby positions
        candidates = []
        base_x = old_point[0]
        base_y = old_point[1]
        
        # Try small perturbations
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_x = base_x + dx
                new_y = base_y + dy
                if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                    candidates.append([new_x, new_y])
        
        # Also try larger jumps
        for _ in range(5):
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            candidates.append([new_x, new_y])
        
        random.shuffle(candidates)
        
        improved = False
        for cand in candidates:
            # Check if candidate collides with other points
            if any(cand[0] == p[0] and cand[1] == p[1] for i, p in enumerate(current) if i != idx):
                continue
            
            # Try the move
            current[idx] = cand[:]
            
            # Quick check: no 5 collinear
            valid = True
            n = len(current)
            for i in range(n):
                for j in range(i + 1, n):
                    key = _line_key(current[i], current[j])
                    count = 1
                    for k in range(n):
                        if k != i and k != j:
                            if _line_key(current[i], current[k]) == key:
                                count += 1
                    if count >= 5:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                current[idx] = old_point[:]
                continue
            
            new_score = _count_4point_lines(current)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if new_score > best_score:
                    best_score = new_score
                    best = [p[:] for p in current]
                break
            else:
                current[idx] = old_point[:]
        
        if not improved:
            # No improvement in this iteration, continue
            pass
    
    return best, best_score

def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_points = None
    best_score = -1
    
    # Try multiple initial configurations
    initializers = [
        _generate_initial_5x5,
        _generate_initial_6x5,
        _generate_initial_4x4_scaled,
    ]
    
    for init_func in initializers:
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            points = init_func()
            points, score = _local_search(points, max_iter=2000, deadline=deadline)
            
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in points]
        
        if time.time() > deadline:
            break
    
    if best_points is None:
        # Fallback: 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I have a syntax error in `_generate_initial_6x5`. Let me fix that.

