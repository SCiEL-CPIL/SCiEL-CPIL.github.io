import time
import itertools
import random
from fractions import Fraction

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    ordinary = 0
    # For each pair, check if the line contains exactly 2 points
    # But we need to avoid double counting - each line with exactly 2 points is counted once
    # Better: for each pair, determine the line, then count how many points are on it
    # Use a set of line identifiers to avoid recounting
    
    # For each pair (i, j), compute the line, then count all points on that line
    # A line is "ordinary" if it contains exactly 2 points
    # We need to count each ordinary line exactly once
    
    # To identify a line uniquely: use (A, B, C) where A*x + B*y + C = 0, normalized
    # For integer points, we can use the direction vector and a point
    
    seen_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Direction vector
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the line: use (A, B, C) with gcd normalization
            # Line: (y2-y1)*x + (x1-x2)*y + (x2*y1 - x1*y2) = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # Normalize by gcd
            g = abs(A) if A != 0 else (abs(B) if B != 0 else abs(C))
            if g == 0:
                continue
            # Compute gcd of all three
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            g = gcd(gcd(A, B), C) if (A != 0 or B != 0) else abs(C)
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            # Ensure canonical form: first nonzero of (A,B) is positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            line_key = (A, B, C)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            # Count points on this line
            cnt = 0
            for k in range(n):
                x, y = points[k]
                if A * x + B * y + C == 0:
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with a 3x4 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    # Try a few different initial configurations
    configs = []
    # 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    # 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    # Other grids
    for w in range(2, 6):
        for h in range(2, 6):
            if w * h == 12:
                configs.append([[x, y] for x in range(w) for y in range(h)])
    
    for config in configs:
        score = count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
    
    # Local search: try moving each point to nearby positions
    random.seed(42)
    iterations = 0
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            x0, y0 = best_points[i]
            # Try moving to nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check distinctness
                    new_points = [list(p) for p in best_points]
                    new_points[i] = [nx, ny]
                    if len(set(tuple(p) for p in new_points)) != 12:
                        continue
                    # Check not all collinear
                    if all((new_points[j][0] - new_points[0][0]) * (new_points[1][1] - new_points[0][1]) == 
                           (new_points[1][0] - new_points[0][0]) * (new_points[j][1] - new_points[0][1]) 
                           for j in range(12)):
                        continue
                    score = count_ordinary_lines(new_points)
                    if score < best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
        if not improved:
            # Try larger moves
            for i in range(12):
                if time.time() > deadline:
                    break
                x0, y0 = best_points[i]
                for _ in range(5):
                    dx = random.randint(-10, 10)
                    dy = random.randint(-10, 10)
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    new_points = [list(p) for p in best_points]
                    new_points[i] = [nx, ny]
                    if len(set(tuple(p) for p in new_points)) != 12:
                        continue
                    if all((new_points[j][0] - new_points[0][0]) * (new_points[1][1] - new_points[0][1]) == 
                           (new_points[1][0] - new_points[0][0]) * (new_points[j][1] - new_points[0][1]) 
                           for j in range(12)):
                        continue
                    score = count_ordinary_lines(new_points)
                    if score < best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
        if not improved:
            break
        iterations += 1
    
    return [list(p) for p in best_points]
