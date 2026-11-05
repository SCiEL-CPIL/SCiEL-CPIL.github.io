import random
import time
from itertools import combinations

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    Points are a list of [x, y] integer coordinates.
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Map from line key to set of point indices on that line
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for Ax + By + C = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
            
            # Normalize the line equation to ensure a unique key
            # Divide by GCD of A, B, C if they are integers
            # Since points are integers, A, B, C are integers.
            # We need to handle signs and scaling.
            
            # Find GCD of absolute values
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Normalize sign: make the first non-zero coefficient positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary_count = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin before 120s limit
    
    best_score = float('inf')
    best_points = None
    
    def try_update(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return
        if len(set(map(tuple, pts))) != 12:
            return
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return
        
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            
    # 1. Initial Candidates
    candidates = []
    
    # 3x4 Grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(grid_3x4)
    
    # 4x3 Grid
    grid_4x3 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(grid_4x3)
    
    # Two lines of 6 (intersecting at origin, but distinct points)
    # L1: y=0, x=1..6
    # L2: x=0, y=1..6
    two_lines = [(i, 0) for i in range(1, 7)] + [(0, i) for i in range(1, 7)]
    candidates.append(two_lines)
    
    # Try to find a good starting point
    for pts in candidates:
        try_update(pts)
        
    if best_points is None:
        # Fallback random start
        best_points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
        # Ensure distinct
        seen = set()
        i = 0
        while len(seen) < 12:
            p = best_points[i]
            if (p[0], p[1]) in seen:
                p[0] += 1
            else:
                seen.add((p[0], p[1]))
            i += 1
        try_update(best_points)
        best_score = count_ordinary_lines(best_points)
        
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # 2. Local Search: Intersection-based moves
    # For each point, try moving it to the intersection of two lines defined by other points
    
    max_iterations = 2000
    no_improve_count = 0
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Get all lines defined by pairs of OTHER points
        # We want to move point idx to an intersection of two such lines
        # To be efficient, we can just pick 2 random pairs of other points and find their intersection
        
        # Select 4 distinct indices from the other 11
        other_indices = [j for j in range(12) if j != idx]
        if len(other_indices) < 4:
            continue
            
        # Pick 4 random distinct indices
        try:
            j1, j2, j3, j4 = random.sample(other_indices, 4)
        except ValueError:
            continue
            
        x1, y1 = current_points[j1]
        x2, y2 = current_points[j2]
        x3, y3 = current_points[j3]
        x4, y4 = current_points[j4]
        
        # Line 1: through (x1,y1) and (x2,y2)
        A1 = y1 - y2
        B1 = x2 - x1
        C1 = x1 * y2 - x2 * y1
        
        # Line 2: through (x3,y3) and (x4,y4)
        A2 = y3 - y4
        B2 = x4 - x3
        C2 = x3 * y4 - x4 * y3
        
        # Intersection of A1*x + B1*y + C1 = 0 and A2*x + B2*y + C2 = 0
        det = A1 * B2 - A2 * B1
        if det == 0:
            # Parallel lines, no intersection
            continue
            
        x_num = B1 * C2 - B2 * C1
        y_num = A2 * C1 - A1 * C2
        
        # Check if intersection is integer
        if x_num % det != 0 or y_num % det != 0:
            continue
            
        new_x = x_num // det
        new_y = y_num // det
        
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check if same as old position
        if new_x == old_x and new_y == old_y:
            continue
            
        # Check for collision with other points
        collision = False
        for j in range(12):
            if j != idx:
                if current_points[j][0] == new_x and current_points[j][1] == new_y:
                    collision = True
                    break
        if collision:
            continue
            
        # Try the move
        current_points[idx] = [new_x, new_y]
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [list(p) for p in current_points]
            no_improve_count = 0
        else:
            # Revert
            current_points[idx] = [old_x, old_y]
            no_improve_count += 1
            
        # If we've had many non-improving moves, try a random jump to escape local minima
        if no_improve_count > 50:
            # Random jump: move a random point to a random location
            idx = random.randint(0, 11)
            old_x, old_y = current_points[idx]
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            
            if not (new_x == old_x and new_y == old_y):
                collision = False
                for j in range(12):
                    if j != idx:
                        if current_points[j][0] == new_x and current_points[j][1] == new_y:
                            collision = True
                            break
                if not collision:
                    current_points[idx] = [new_x, new_y]
                    new_score = count_ordinary_lines(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_points = [list(p) for p in current_points]
                        no_improve_count = 0
                    else:
                        current_points[idx] = [old_x, old_y]
            
            no_improve_count = 0

    return best_points

# EVOLVE-BLOCK-START
# The code above is the complete program.
# The entry point is construct()
# EVOLVE-BLOCK-END
