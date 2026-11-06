import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    g = _gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Line: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _count_points_on_line(line_key, points):
    """Count how many points lie on the given line."""
    if line_key is None:
        return 0
    dx, dy, c = line_key
    count = 0
    for x, y in points:
        if dy * x - dx * y == c:
            count += 1
    return count

def _score_and_lines(points):
    """
    Returns (score, line_counts_dict).
    Score is number of lines with exactly 4 points.
    Returns -1 if any line has 5+ points.
    """
    n = len(points)
    if n != 16:
        return -1, {}
    
    line_counts = {}
    # Generate all unique lines defined by pairs
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_counts:
                line_counts[key] = _count_points_on_line(key, points)
            
            # Early exit if 5+ collinear
            if line_counts[key] >= 5:
                return -1, line_counts

    count4 = 0
    for key, cnt in line_counts.items():
        if cnt == 4:
            count4 += 1
    return count4, line_counts

def _is_valid(points):
    """Check if points are valid: 16 distinct, |x|,|y|<=1000, no 5 collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    
    score, _ = _score_and_lines(points)
    return score >= 0

def _local_search(points, deadline, rng):
    """
    Improve points via local search with line-completion heuristics.
    """
    best = [list(p) for p in points]
    best_score, _ = _score_and_lines(best)
    if best_score < 0:
        best_score = 0
    
    # Pre-compute some candidate moves? No, dynamic is better.
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try line-completion moves
        # Find lines with exactly 3 points.
        _, line_counts = _score_and_lines(best)
        if _score_and_lines(best)[0] < 0:
            # If invalid, try to fix by moving a point off the 5-collinear line
            # This is complex, so we rely on random restarts for invalid states.
            pass
            
        # Identify 3-point lines
        three_point_lines = []
        for key, cnt in line_counts.items():
            if cnt == 3:
                three_point_lines.append(key)
        
        if three_point_lines:
            # For each 3-point line, try to add a 4th point
            # We need to find a point P on the line such that P is not in best,
            # and P is within bounds.
            # Then, we move one of the existing points (the one with least impact) to P?
            # Or if P is empty, we move a "lonely" point to P.
            
            # To keep it simple:
            # For each 3-line, calculate the 4th integer point if it exists on the line segment extension.
            # Actually, any integer point on the line works.
            # Let's just try moving any point to any valid position on a 3-point line.
            
            for key in three_point_lines:
                if time.time() > deadline:
                    break
                dx, dy, c = key
                # Find integer points on this line
                # dy*x - dx*y = c
                # We can parametrize.
                # But simpler: Just try moving each point to a random position on this line?
                # That's expensive.
                
                # Alternative: Check if there's a "hole" in a potential 4-point line.
                # This is hard to define generically without enumerating all 4-point lines.
                
                # Let's stick to: Try moving each point to a nearby position.
                pass

        # 2. Standard local search: Move one point to a nearby location
        for idx in range(16):
            if time.time() > deadline:
                break
            x, y = best[idx]
            
            # Small moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_points = [list(p) for p in best]
                    new_points[idx] = [nx, ny]
                    if len(set(tuple(p) for p in new_points)) != 16:
                        continue
                    s, _ = _score_and_lines(new_points)
                    if s > best_score:
                        best = new_points
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # 3. Larger random moves if stuck
        if not improved:
            # Try a few random teleports
            for _ in range(20):
                if time.time() > deadline:
                    break
                idx = rng.randint(0, 15)
                nx = rng.randint(-1000, 1000)
                ny = rng.randint(-1000, 1000)
                new_points = [list(p) for p in best]
                new_points[idx] = [nx, ny]
                if len(set(tuple(p) for p in new_points)) != 16:
                    continue
                s, _ = _score_and_lines(new_points)
                if s > best_score:
                    best = new_points
                    best_score = s
                    improved = True
                    break
            
            if not improved:
                # Accept a worse move with some probability (Simulated Annealing step)
                # Or just break and restart
                # Let's try a random move that doesn't decrease score too much
                for _ in range(10):
                    idx = rng.randint(0, 15)
                    nx = best[idx][0] + rng.randint(-20, 20)
                    ny = best[idx][1] + rng.randint(-20, 20)
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_points = [list(p) for p in best]
                    new_points[idx] = [nx, ny]
                    if len(set(tuple(p) for p in new_points)) != 16:
                        continue
                    s, _ = _score_and_lines(new_points)
                    if s >= best_score - 1: # Allow slight degradation
                        best = new_points
                        best_score = s
                        improved = True
                        break
                if not improved:
                    break # Restart needed

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    rng = random.Random(42)
    
    # Initial configuration: 4x4 grid
    # This gives 10 lines.
    init_points = [[x, y] for x in range(4) for y in range(4)]
    
    best_global = [list(p) for p in init_points]
    best_global_score, _ = _score_and_lines(best_global)
    if best_global_score < 0:
        best_global_score = 0

    # Run multiple restarts
    while time.time() < deadline:
        # 1. Start from current best or a perturbed version
        current = [list(p) for p in best_global]
        
        # Perturb slightly to escape local optima
        for _ in range(rng.randint(1, 5)):
            idx = rng.randint(0, 15)
            current[idx] = [current[idx][0] + rng.randint(-5, 5), 
                            current[idx][1] + rng.randint(-5, 5)]
        
        # Ensure validity (bounds and distinctness)
        valid = True
        seen = set()
        for i, (x, y) in enumerate(current):
            if abs(x) > 1000 or abs(y) > 1000:
                current[i] = [0, 0] # Reset to safe, will be fixed by search
                valid = False
            if (x, y) in seen:
                current[i] = [rng.randint(-100, 100), rng.randint(-100, 100)]
                valid = False
            seen.add((x, y))
            
        # If we messed up too much, just use the 4x4 grid again
        if not valid or len(set(tuple(p) for p in current)) != 16:
            current = [list(p) for p in init_points]

        # Local search
        result = _local_search(current, deadline, rng)
        s, _ = _score_and_lines(result)
        
        if s > best_global_score:
            best_global = result
            best_global_score = s
            # Print for debugging if needed, but not required.
            # print(f"New best: {best_global_score}")
            
            # If we hit the theoretical max, stop early
            if best_global_score >= 15:
                break
        else:
            # If no improvement, try a completely random restart occasionally
            if rng.random() < 0.1:
                # Random restart
                current = [[rng.randint(-1000, 1000), rng.randint(-1000, 1000)] for _ in range(16)]
                # Ensure distinctness
                for i in range(16):
                    while (current[i][0], current[i][1]) in set(tuple(p) for p in current[:i]):
                        current[i] = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                result = _local_search(current, deadline, rng)
                s, _ = _score_and_lines(result)
                if s > best_global_score:
                    best_global = result
                    best_global_score = s

    return best_global

# EVOLVE-BLOCK-END
