import time
import random
import math
from collections import defaultdict

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = defaultdict(int)
    
    # Generate all pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients Ax + By = C
            A = y2 - y1
            B = x1 - x2
            C = x1 * y2 - x2 * y1
            
            # Normalize (A, B, C)
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            # Canonical sign: A > 0, or A == 0 and B > 0
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            line_counts[(A, B, C)] += 1
    
    ordinary_count = 0
    for count in line_counts.values():
        disc = 1 + 8 * count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k == 2:
                ordinary_count += 1
                
    return ordinary_count

def generate_structured_candidates():
    """Generate known structured configurations likely to have few ordinary lines."""
    candidates = []
    
    # 1. Points on a cubic curve: y = x^3 mod p for suitable p, or parametric cubic
    # Try points on y = x^3 for x in a set, plus some offset
    # Actually, let's try various small structured sets
    
    # 2. Grid-based with high collinearity: 3x4 grid has many ordinary lines
    # Better: points arranged so many lines have 3+ points
    
    # 3. Use a known configuration: 12 points as vertices/intersections of a complete quadrilateral
    # or points on a conic + some extra
    
    # Let's try: 12 points on the curve y = x^2 for x in {-3,-2,-1,0,1,2,3,4,5,6,7,8}
    # But that's a parabola - all points are on one line? No, they're on a curve.
    # Lines through pairs of points on a parabola: each pair defines a unique line,
    # but 3 points on a parabola are never collinear (unless degenerate).
    # So a parabola gives C(12,2) = 66 ordinary lines. Bad.
    
    # Better: use a configuration with many 3-point lines.
    # Classic: 9 points in 3x3 grid has 8 lines with 3 points (3 rows, 3 cols, 2 diagonals)
    # For 12 points, we can extend this.
    
    # Try: 12 points = 3x4 grid but with some modification
    # Or: two parallel lines of 4 points each + 4 more points
    
    # Known good configuration for 12 points: 
    # Place points at (i, j) for i in {0,1,2,3}, j in {0,1,2,3} minus 4 points, plus 4 others
    
    # Let's just generate several candidates and let the search improve them
    
    # Candidate 1: 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])
    
    # Candidate 2: 4x3 grid  
    candidates.append([[x, y] for x in range(3) for y in range(4)])
    
    # Candidate 3: Points on two parallel lines
    # Line y=0: x = 0,1,2,3,4,5
    # Line y=1: x = 0,1,2,3,4,5
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Candidate 4: Points forming a hexagon-like structure
    # 6 points on a circle (approximated by integers) + 6 more
    # Use: (±1,0), (0,±1), (±1,±1) scaled
    pts = []
    for angle in range(12):
        x = 10 * round(math.cos(angle * math.pi / 6))
        y = 10 * round(math.sin(angle * math.pi / 6))
        if (x, y) not in [(p[0], p[1]) for p in pts]:
            pts.append([x, y])
    if len(pts) == 12:
        candidates.append(pts)
    
    # Candidate 5: 3 groups of 4 points in a line, lines in different directions
    # Group 1: horizontal line y=0, x=0,1,2,3
    # Group 2: vertical line x=5, y=0,1,2,3
    # Group 3: diagonal, points (6,0),(7,1),(8,2),(9,3)
    # Plus 4 more points to make 12... this is getting complex.
    
    # Let's try a simpler structured approach:
    # 12 points = 4 points on each of 3 parallel lines
    candidates.append([[x, 0] for x in range(4)] + 
                      [[x, 1] for x in range(4)] + 
                      [[x, 2] for x in range(4)])
    
    return candidates

def local_search_improved(points, deadline, max_iter=100000):
    """Improved local search with better move generation."""
    current_points = [p[:] for p in points]
    current_score = count_ordinary_lines(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Pick a random point to move
        idx = random.randint(0, 11)
        orig_x, orig_y = current_points[idx]
        
        # Generate move: try to place point on a line that already has 2+ other points
        # This reduces ordinary lines
        move_found = False
        
        # Strategy 1: Pick another point, then a third, and try to place on their line
        if random.random() < 0.7:
            # Pick two other points
            other_indices = [i for i in range(12) if i != idx]
            i1 = random.choice(other_indices)
            i2 = random.choice([i for i in other_indices if i != i1])
            
            x1, y1 = current_points[i1]
            x2, y2 = current_points[i2]
            
            # Line through (x1,y1) and (x2,y2): find integer points on it
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g > 0:
                dx_unit = dx // g
                dy_unit = dy // g
                
                # Try several positions along this line
                for t in range(-20, 21):
                    new_x = x1 + t * dx_unit
                    new_y = y1 + t * dy_unit
                    
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check distinctness
                    is_distinct = True
                    for i, p in enumerate(current_points):
                        if i == idx:
                            continue
                        if p[0] == new_x and p[1] == new_y:
                            is_distinct = False
                            break
                    
                    if not is_distinct:
                        continue
                    
                    new_points = [p[:] for p in current_points]
                    new_points[idx] = [new_x, new_y]
                    new_score = count_ordinary_lines(new_points)
                    
                    if new_score < current_score:
                        current_points = new_points
                        current_score = new_score
                        move_found = True
                        no_improve_count = 0
                        break
                    elif new_score == current_score and random.random() < 0.3:
                        current_points = new_points
                        move_found = True
                        break
        
        # Strategy 2: Random small move
        if not move_found:
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            new_x = orig_x + dx
            new_y = orig_y + dy
            
            if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                is_distinct = True
                for i, p in enumerate(current_points):
                    if i == idx:
                        continue
                    if p[0] == new_x and p[1] == new_y:
                        is_distinct = False
                        break
                
                if is_distinct:
                    new_points = [p[:] for p in current_points]
                    new_points[idx] = [new_x, new_y]
                    new_score = count_ordinary_lines(new_points)
                    
                    if new_score < current_score:
                        current_points = new_points
                        current_score = new_score
                        no_improve_count = 0
                    elif new_score == current_score and random.random() < 0.5:
                        current_points = new_points
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            if best_score == 6:
                return best_points
        else:
            no_improve_count += 1
        
        # Reset if stuck too long
        if no_improve_count > 5000:
            # Try a larger random move
            idx = random.randint(0, 11)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                is_distinct = True
                for i, p in enumerate(current_points):
                    if i == idx:
                        continue
                    if p[0] == new_x and p[1] == new_y:
                        is_distinct = False
                        break
                if is_distinct:
                    current_points[idx] = [new_x, new_y]
                    current_score = count_ordinary_lines(current_points)
            no_improve_count = 0
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try structured candidates first
    candidates = generate_structured_candidates()
    
    best_points = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(cand) != 12:
            continue
        # Check distinctness
        seen = set()
        valid = True
        for p in cand:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid:
            continue
        
        score = count_ordinary_lines(cand)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cand]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)
    
    # If we already hit 6, return immediately
    if best_score == 6:
        return best_points
    
    # Run local search from the best candidate
    time_left = deadline - time.time()
    if time_left > 5:
        result = local_search_improved(best_points, deadline, max_iter=200000)
        if result is not None:
            score = count_ordinary_lines(result)
            if score < best_score:
                best_score = score
                best_points = result
    
    return best_points
