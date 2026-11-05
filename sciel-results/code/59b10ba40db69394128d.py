"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _collinear3(p1, p2, p3):
    """Check if three points are collinear."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if not _collinear3(points[i], points[j], points[k]):
                    continue
                for l in range(k + 1, n):
                    if _collinear3(points[i], points[j], points[l]):
                        return True
    return False

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(points[i], points[j], points[k]):
                    triples += 1
    return triples

def _max_3free_subset(points, deadline=None):
    """Branch and bound to find the largest subset with no three collinear."""
    n = len(points)
    best = [0]
    
    # Precompute collinearity
    collinear = [[[False] * n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(points[i], points[j], points[k]):
                    collinear[i][j][k] = True
                    collinear[j][i][k] = True
                    collinear[k][i][j] = True
                    collinear[i][k][j] = True
                    collinear[j][k][i] = True
                    collinear[k][j][i] = True
    
    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        if current_size + (n - idx) <= best[0]:
            return
        
        # Try excluding idx first
        dfs(idx + 1, current, current_size)
        
        # Try including idx
        can_add = True
        for i, ci in enumerate(current):
            for j in range(i + 1, len(current)):
                cj = current[j]
                if collinear[idx][ci][cj]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            current.append(idx)
            dfs(idx + 1, current, current_size + 1)
            current.pop()
    
    dfs(0, [], 0)
    return best[0]

def _deterministic_base():
    """
    Deterministic construction using a structured approach.
    Use points on a curve with controlled perturbations to create
    many 3-point lines while avoiding 4-collinear.
    
    Strategy: Start with points on y = x^2 (parabola) which has no 3 collinear,
    then systematically move points to create 3-point lines.
    """
    # Start with 20 points on a parabola: (x, x^2) for x in some range
    # This gives no 3 collinear, so we need to perturb
    points = []
    for x in range(-10, 10):
        points.append([x, x * x])
    
    return points

def _create_3lines_from_parabola():
    """
    Take the parabola and create 3-point lines by moving points.
    For each pair of points, find a third point that would create a 3-line,
    and move an existing point there if valid.
    """
    points = _deterministic_base()
    n = len(points)
    
    # Try to create 3-point lines by moving points
    for iteration in range(100):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                # Line through points[i] and points[j]
                p1, p2 = points[i], points[j]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Find integer points on this line
                if dx == 0:
                    # Vertical line
                    x = p1[0]
                    for y in range(-100, 101):
                        if (x, y) == tuple(p1) or (x, y) == tuple(p2):
                            continue
                        # Check if this creates a valid 3-line
                        new_points = points[:]
                        # Find which point to move: move a point that's not on this line
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if not _collinear3(p1, p2, points[k]):
                                # Try moving point k to (x, y)
                                old_pos = points[k][:]
                                points[k] = [x, y]
                                if len(set(tuple(p) for p in points)) == 20 and not _has_four_collinear(points):
                                    # Check if this creates a 3-line
                                    if _collinear3(p1, p2, [x, y]):
                                        improved = True
                                        break
                                points[k] = old_pos
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                else:
                    # Non-vertical line: parametric form
                    # p(t) = p1 + t * (dx, dy)
                    # For integer points, t must be rational
                    g = gcd(abs(dx), abs(dy))
                    step_x = dx // g
                    step_y = dy // g
                    
                    # Points on line: p1 + t * (step_x, step_y) for integer t
                    # Find t values that give integer coordinates in range
                    for t in range(-200, 201):
                        x = p1[0] + t * step_x
                        y = p1[1] + t * step_y
                        if x == p1[0] and y == p1[1]:
                            continue
                        if x == p2[0] and y == p2[1]:
                            continue
                        if -100 <= x <= 100 and -100 <= y <= 100:
                            # Try to move a point to (x, y)
                            for k in range(n):
                                if k == i or k == j:
                                    continue
                                if not _collinear3(p1, p2, points[k]):
                                    old_pos = points[k][:]
                                    points[k] = [x, y]
                                    if len(set(tuple(p) for p in points)) == 20 and not _has_four_collinear(points):
                                        improved = True
                                        break
                                    points[k] = old_pos
                                if improved:
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
            if not improved:
                break
    
    return points

def _local_search_3line(points, deadline):
    """
    Local search that specifically tries to create 3-point lines.
    For each point, try moving it to create a new 3-point line with two existing points.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    
    def get_line(p1, p2):
        """Return (A, B, C) for line Ax + By + C = 0, normalized."""
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        if A == 0 and B == 0:
            return (0, 0, 0)
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    def point_on_line(p, line):
        A, B, C = line
        return A * p[0] + B * p[1] + C == 0
    
    while time.time() < deadline:
        improved = False
        n = len(points)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            orig = points[i][:]
            
            # Try to move point i to create a 3-point line with two other points
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    
                    line_jk = get_line(points[j], points[k])
                    if point_on_line(orig, line_jk):
                        continue
                    
                    A, B, C = line_jk
                    candidates_on_line = []
                    
                    if B != 0:
                        for x in range(-100, 101):
                            num = -A * x - C
                            if num % B == 0:
                                y = num // B
                                if -100 <= y <= 100:
                                    candidates_on_line.append([x, y])
                    elif A != 0:
                        for y in range(-100, 101):
                            num = -B * y - C
                            if num % A == 0:
                                x = num // A
                                if -100 <= x <= 100:
                                    candidates_on_line.append([x, y])
                    
                    # Try candidates
                    for new_pos in candidates_on_line:
                        if new_pos in [points[m] for m in range(n) if m != i]:
                            continue
                        
                        old_pos = points[i][:]
                        points[i] = new_pos[:]
                        
                        if len(set(tuple(p) for p in points)) != 20:
                            points[i] = old_pos
                            continue
                        if _has_four_collinear(points):
                            points[i] = old_pos
                            continue
                        
                        new_triples = _count_collinear_triples(points)
                        if new_triples > current_triples:
                            current_triples = new_triples
                            improved = True
                            break
                        else:
                            points[i] = old_pos
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return points

def _simulated_annealing(points, deadline):
    """
    Simulated annealing to minimize the 3-free subset size.
    We use the number of collinear triples as a proxy: more triples = smaller 3-free subset.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    best_points = [p[:] for p in points]
    best_triples = current_triples
    
    start_time = time.time()
    t = 10.0
    cooling = 0.99
    
    while time.time() < deadline and t > 0.1:
        # Pick a random point to move
        i = random.randint(0, 19)
        old_pos = points[i][:]
        
        # Try a random new position
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        new_pos = [new_x, new_y]
        
        # Check if position is already used
        if new_pos in [points[m] for m in range(20) if m != i]:
            t *= cooling
            continue
        
        points[i] = new_pos
        
        # Check validity
        if len(set(tuple(p) for p in points)) != 20 or _has_four_collinear(points):
            points[i] = old_pos
            t *= cooling
            continue
        
        new_triples = _count_collinear_triples(points)
        delta = new_triples - current_triples
        
        # Accept if better, or with probability exp(delta / t) if worse
        if delta >= 0 or random.random() < 2.718281828 ** (delta / t):
            current_triples = new_triples
            if current_triples > best_triples:
                best_triples = current_triples
                best_points = [p[:] for p in points]
        else:
            points[i] = old_pos
        
        t *= cooling
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_triples = -1
    
    # Phase 1: Deterministic construction
    points = _create_3lines_from_parabola()
    if points and len(set(tuple(p) for p in points)) == 20 and not _has_four_collinear(points):
        triples = _count_collinear_triples(points)
        best_triples = triples
        best_points = [p[:] for p in points]
    
    # Phase 2: Local search to create more 3-lines
    if best_points and time.time() < deadline - 30:
        improved = _local_search_3line(best_points, deadline - 25)
        if len(set(tuple(p) for p in improved)) == 20 and not _has_four_collinear(improved):
            improved_triples = _count_collinear_triples(improved)
            if improved_triples > best_triples:
                best_triples = improved_triples
                best_points = [p[:] for p in improved]
    
    # Phase 3: Simulated annealing for further improvement
    if best_points and time.time() < deadline - 10:
        sa_result = _simulated_annealing(best_points, deadline - 5)
        if len(set(tuple(p) for p in sa_result)) == 20 and not _has_four_collinear(sa_result):
            sa_triples = _count_collinear_triples(sa_result)
            if sa_triples > best_triples:
                best_triples = sa_triples
                best_points = [p[:] for p in sa_result]
    
    # Return the best points found
    if best_points:
        return best_points
    else:
        # Fallback: parabola
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
