import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper: Normalize a line defined by two points to a canonical key
    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return None
        # Normalize direction vector
        g = math.gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g
        # Ensure canonical sign (e.g., dx > 0 or dx == 0 and dy > 0)
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        
        # The line is defined by its direction (dx, dy) and its offset from origin.
        # Offset is the cross product of the direction vector and a point on the line.
        # For a line with direction (dx, dy), the normal is (-dy, dx).
        # Constant c = -dy*x + dx*y
        c = -dy * x1 + dx * y1
        return (dx, dy, c)

    def count_4_lines(points):
        """
        Counts lines containing exactly 4 points.
        Also returns the number of lines containing >= 5 points (for penalty).
        """
        line_counts = {}
        # n=16, so C(16,2) = 120 pairs. Very fast.
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                k = line_key(points[i], points[j])
                if k:
                    line_counts[k] = line_counts.get(k, 0) + 1
        
        exact_4 = 0
        bad_lines = 0
        for k, count in line_counts.items():
            if count == 4:
                exact_4 += 1
            elif count > 4:
                bad_lines += 1
                # Heavy penalty for any line with >4 points
                # We want to maximize exact_4, but bad_lines must be 0.
                # We can return a tuple (exact_4, -bad_lines) or just exact_4 if we ensure validity.
                # For local search, it's better to have a single score.
                # Let's use: score = exact_4 * 1000 - bad_lines * 100000
                # This way, avoiding bad lines is prioritized, then maximizing 4-lines.
        
        return exact_4, bad_lines

    def score(points):
        e4, bad = count_4_lines(points)
        # If there are bad lines, the score is very low
        if bad > 0:
            return -1000000 + e4
        return e4

    def is_valid(points):
        # Check distinctness and bounds
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
            if not (-1000 <= p[0] <= 1000 and -1000 <= p[1] <= 1000):
                return False
        _, bad = count_4_lines(points)
        return bad == 0

    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    def random_config():
        pts = []
        while len(pts) < 16:
            p = random_point()
            if p not in pts:
                pts.append(p)
        return pts

    best_score = -1
    best_points = None
    
    # Try a few random starts
    for _ in range(5):
        if time.time() > deadline:
            break
            
        current_points = random_config()
        current_score = score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        # Simulated Annealing / Local Search
        temp = 50.0
        cooling = 0.99
        min_temp = 0.1
        
        while temp > min_temp:
            if time.time() > deadline:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 15)
            old_p = current_points[idx][:]
            
            # Try moving it to a nearby random location
            # Step size can be adaptive
            step = random.randint(1, 20)
            new_p = [old_p[0] + random.randint(-step, step), 
                     old_p[1] + random.randint(-step, step)]
            
            # Check bounds
            if not (-1000 <= new_p[0] <= 1000 and -1000 <= new_p[1] <= 1000):
                continue
                
            # Check distinctness
            if any(new_p == p for i, p in enumerate(current_points) if i != idx):
                continue
                
            # Apply move
            current_points[idx] = new_p
            new_score = score(current_points)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(delta/temp) if worse
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[idx] = old_p
                
            temp *= cooling
            
    # If we haven't found a valid solution, or if best is still bad, try a grid-based fallback
    # The 4x4 grid is a known valid configuration with score 6 (4 rows, 4 cols, 2 diags? No, 4x4 grid has 4 rows, 4 cols, 2 main diags = 10 lines of 4? Wait.
    # In a 4x4 grid:
    # Rows: 4 lines of 4.
    # Cols: 4 lines of 4.
    # Diagonals: 2 lines of 4.
    # Other diagonals of slope 1 or -1 with length 4?
    # In 4x4, the main diagonals are length 4.
    # Are there other lines of 4?
    # Slope 1: (0,0)-(1,1)-(2,2)-(3,3). (0,1)-(1,2)-(2,3) is len 3.
    # So total 10 lines of exactly 4?
    # Let's check the seed score. Seed returns 4x4 grid. Score 1.0 implies value 10? Or maybe 15?
    # If seed score is 1.0 and it's the baseline, and the task says "real value 15 is an upper reference",
    # it's possible the 4x4 grid has 10 lines.
    # Let's ensure our local search can at least find the 4x4 grid or better.
    
    # Fallback: 4x4 Grid
    grid_points = [[x, y] for x in range(4) for y in range(4)]
    grid_score = score(grid_points)
    
    if best_points is None or best_score < grid_score:
        best_points = grid_points
        best_score = grid_score

    return best_points
# EVOLVE-BLOCK-END
