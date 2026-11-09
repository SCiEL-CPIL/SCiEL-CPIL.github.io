"""Rewritten program: combinatorial design + greedy local search for 12 rational points with few ordinary lines."""
import time
import random
from math import gcd

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Map each line (represented by a canonical (a,b,c) of ax+by+c=0) to the set of point indices on it.
    line_points = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            # Line through i and j: (y - yi)*(xj - xi) = (x - xi)*(yj - yi)
            # => (yj - yi)*x - (xj - xi)*y + (xj - xi)*yi - (yj - yi)*xi = 0
            a = yj - yi
            b = -(xj - xi)
            c = (xj - xi) * yi - (yj - yi) * xi
            # Canonicalize the line equation (a, b, c) so that the same line has the same key.
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
            if g:
                a //= g
                b //= g
                c //= g
            # Make (a, b) have a consistent sign
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    ordinary = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, n):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _evaluate(points):
    """Return the number of ordinary lines, or None if invalid."""
    if len(points) != 12:
        return None
    # Check distinctness
    seen = set()
    for p in points:
        t = (p[0], p[1])
        if t in seen:
            return None
        seen.add(t)
    if not _not_all_collinear(points):
        return None
    return _count_ordinary_lines(points)

def _generate_pappus_based():
    """
    Generate 12 points using a structure inspired by Pappus's hexagon theorem.
    
    The Pappus configuration has 9 points on 9 lines (3 points per line).
    We extend this by adding 3 more points strategically to minimize ordinary lines.
    
    Strategy: Place 9 points in a Pappus-like arrangement, then add 3 points
    that each lie on 2+ existing lines to minimize new ordinary lines.
    """
    # Use a 3x3 grid perturbed to create the Pappus-like structure
    # Then add 3 points on existing lines
    
    # Base 3x3 grid
    base = [[x, y] for x in range(3) for y in range(3)]
    
    # The 3x3 grid has 8 lines with 3 points: 3 rows, 3 cols, 2 diagonals
    # That gives us 9 lines with 3 points each, covering all 9 points
    
    # Now add 3 more points. We want them to lie on existing 3-point lines
    # to avoid creating new ordinary lines.
    
    # Lines in 3x3 grid:
    # Rows: y=0, y=1, y=2 (each has 3 points)
    # Cols: x=0, x=1, x=2 (each has 3 points)  
    # Diagonals: y=x, y=-x+2 (each has 3 points)
    
    # Let's extend the grid to create more collinearities
    # Add points at (3,0), (0,3), (3,3) to extend some lines
    
    # Actually, let's try a different approach: use a 4x3 grid with modifications
    # A 4x3 grid has many 3-point lines but also many 2-point lines
    
    # Better: use specific rational points that create many 3-collinearities
    
    # Try: points on two parallel lines with 6 points each, plus strategic additions
    # Line 1: y=0, points at x = 0,1,2,3,4,5
    # Line 2: y=1, points at x = 0,1,2,3,4,5
    # But this gives only 2 lines with 6 points, and many ordinary lines between them
    
    # Let's use a hexagonal lattice approach with integer coordinates
    # Hexagonal lattice: (i, j) -> (i + j//2, j * sqrt(3)//2) doesn't work for integers
    
    # Alternative: use a "thick" grid structure
    # Place points at:
    # (0,0), (1,0), (2,0), (3,0)  - 4 on y=0
    # (0,1), (1,1), (2,1), (3,1)  - 4 on y=1  
    # (0,2), (1,2), (2,2)         - 3 on y=2
    # (1,3)                       - 1 on y=3
    # Total: 12 points
    
    # But this still has many ordinary lines.
    
    # Let me try a completely different approach: use the fact that we can
    # place multiple points on the same line to reduce ordinary lines.
    
    # Optimal strategy: maximize the number of points on lines with 3+ points.
    # With 12 points, if we can get:
    # - 4 lines with 3 points each (12 point-slots, but points can be on multiple lines)
    # - The key is that each pair of points should share at least one 3+ point line
    
    # Let's try: 3 lines, each with 4 points, arranged in a triangle
    # Line A: y=0, points at x=0,1,2,3
    # Line B: y=x, points at (0,0),(1,1),(2,2),(3,3) - but (0,0) is already on A
    # This creates overlaps which is good!
    
    # Configuration:
    # Points: (0,0), (1,0), (2,0), (3,0)  on y=0
    #         (0,0), (1,1), (2,2), (3,3)  on y=x  [(0,0) shared]
    #         (0,0), (0,1), (0,2), (0,3)  on x=0  [(0,0) shared]
    # Distinct points: (0,0), (1,0), (2,0), (3,0), (1,1), (2,2), (3,3), (0,1), (0,2), (0,3)
    # That's only 10 distinct points. Need 2 more.
    
    # Add (1,2) and (2,1):
    # (1,2) is on line through (0,1) and (2,3)? No, (2,3) doesn't exist.
    # (1,2) is on line through (0,2) and (2,2)? Yes! y=2 line: (0,2),(1,2),(2,2) - 3 points
    # (2,1) is on line through (0,1) and (2,1)? And (1,1)? Line y=1: (0,1),(1,1),(2,1) - 3 points
    
    # Let me list all 12 points:
    pts = [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [2, 1]
    ]
    
    return pts

def _generate_hexagonal_lattice():
    """
    Generate 12 points on a hexagonal lattice pattern using integer coordinates.
    The hexagonal lattice has high symmetry with many 3-point collinearities.
    """
    # Hexagonal lattice in axial coordinates (q, r):
    # Cartesian: x = q + r/2, y = r * sqrt(3)/2
    # To get integer coords, use: x = 2q + r, y = 3r (approximate)
    # Or just use a rectangular grid with hexagonal-like collinearities
    
    # Let's use a 4x3 grid which has good collinear structure
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def _generate_parabola_points():
    """
    Generate points on a parabola y = x^2 and related lines.
    Points on a parabola have the property that lines through pairs
    intersect the parabola at specific points, creating structure.
    """
    # Points on y = x^2: (0,0), (1,1), (2,4), (3,9), (4,16), (5,25)
    # But we need 12 points. Let's use two parabolas or a parabola plus lines.
    
    # Alternative: use points that create many 3-collinearities
    # Place 6 points on one line, 6 on another, with intersections creating more lines
    
    # Line 1: y = 0, points at x = 0, 1, 2, 3, 4, 5
    # Line 2: y = 1, points at x = 0, 1, 2, 3, 4, 5
    # This gives 12 points, 2 lines with 6 points each
    # Ordinary lines: all lines connecting a point on line 1 to a point on line 2
    # except those that pass through 3+ points
    
    # Actually this might have many ordinary lines. Let me think...
    # Line from (a,0) to (b,1): slope = 1/(b-a), equation: y = (x-a)/(b-a)
    # For this to pass through (c,0) or (d,1), we need specific conditions
    
    # Better: use a 3x4 grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    return pts

def _smart_local_search(points, max_iter=100000):
    """
    Smart local search that specifically targets creating 3+ point collinearities.
    For each point, try moving it to a position that lies on a line through
    two other existing points (creating a 3-point line).
    """
    current = [list(p) for p in points]
    current_score = _evaluate(current)
    if current_score is None:
        return current, current_score
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
        
        # Strategy 1: Try to move a point to lie on an existing 2-point line
        # to create a 3-point line
        idx = random.randint(0, 11)
        
        # Find all lines through pairs of other points
        candidate_positions = set()
        for i in range(12):
            if i == idx:
                continue
            for j in range(i + 1, 12):
                if j == idx:
                    continue
                # Line through points i and j
                xi, yi = current[i]
                xj, yj = current[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                
                # Try extending this line to find integer points
                # The line is: (x - xi)*dy = (y - yi)*dx
                # For integer solutions, try small offsets
                for t in range(-5, 6):
                    if t == 0:
                        continue
                    nx = xi + t * dx
                    ny = yi + t * dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        # Check if this position is not already taken
                        pos = (nx, ny)
                        if pos not in [(current[k][0], current[k][1]) for k in range(12) if k != idx]:
                            candidate_positions.add((nx, ny))
        
        if candidate_positions:
            # Pick a random candidate
            nx, ny = random.choice(list(candidate_positions))
            old_point = current[idx]
            current[idx] = [nx, ny]
            new_score = _evaluate(current)
            
            if new_score is not None and new_score <= current_score:
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best = [list(p) for p in current]
            else:
                current[idx] = old_point
        else:
            # Fallback: random small perturbation
            x, y = current[idx]
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            old_point = current[idx]
            current[idx] = [nx, ny]
            new_score = _evaluate(current)
            
            if new_score is not None and new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = [list(p) for p in current]
            else:
                current[idx] = old_point
    
    return best, best_score

def _try_multiple_starts():
    """Try multiple starting configurations and local search from each."""
    best_points = None
    best_score = float('inf')
    
    # Start 1: Pappus-based configuration
    pts1 = _generate_pappus_based()
    p1, s1 = _smart_local_search(pts1, max_iter=50000)
    if s1 is not None and s1 < best_score:
        best_score = s1
        best_points = p1
    
    if _elapsed() > _DEADLINE:
        return best_points, best_score
    
    # Start 2: Hexagonal lattice / 4x3 grid
    pts2 = _generate_hexagonal_lattice()
    p2, s2 = _smart_local_search(pts2, max_iter=50000)
    if s2 is not None and s2 < best_score:
        best_score = s2
        best_points = p2
    
    if _elapsed() > _DEADLINE:
        return best_points, best_score
    
    # Start 3: Random perturbations of the grid
    for restart in range(3):
        if _elapsed() > _DEADLINE:
            break
        random_pts = []
        for x in range(4):
            for y in range(3):
                rx = x + random.randint(-2, 2)
                ry = y + random.randint(-2, 2)
                random_pts.append([rx, ry])
        pr, sr = _smart_local_search(random_pts, max_iter=30000)
        if sr is not None and sr < best_score:
            best_score = sr
            best_points = pr
    
    return best_points, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points, best_score = _try_multiple_starts()
    
    if best_points is None:
        # Fallback: return the 3x4 grid
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
