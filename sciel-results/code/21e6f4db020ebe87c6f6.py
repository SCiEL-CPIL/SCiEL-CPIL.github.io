import time
import random
import math
from fractions import Fraction

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            
            g = gcd(gcd(A, B), C)
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            lines[key] = lines.get(key, 0) + 1
    
    ordinary_count = 0
    for key, count in lines.items():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def is_all_collinear(points):
    if len(points) < 3:
        return False
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return False
    
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x2 - x1) * (yi - y1) != (y2 - y1) * (xi - x1):
            return False
            
    return True

def line_through_two_points(p1, p2):
    """Return (A, B, C) normalized such that A*x + B*y + C = 0, with gcd(A,B,C)=1 and canonical sign."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    g = gcd(gcd(A, B), C)
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def get_integer_point_on_line(p1, p2, t_range=(-10, 10)):
    """
    Find a random integer point on the line defined by p1 and p2,
    parameterized by p1 + t * (p2 - p1).
    Returns [x, y] or None if no valid point found.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return None
        
    g = gcd(abs(dx), abs(dy))
    
    # The step size for integer points is g
    # We want p1 + k * (dx/g, dy/g)
    step_x = dx // g
    step_y = dy // g
    
    # Try random k in t_range
    for _ in range(10):
        k = random.randint(t_range[0], t_range[1])
        if k == 0: continue # p1 itself
        if k == 1: continue # p2 itself (though distinctness check handles it, skip for efficiency)
        
        nx = x1 + k * step_x
        ny = y1 + k * step_y
        
        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
            return [nx, ny]
            
    return None

def local_search(points, max_iters=1000):
    """
    Greedy local search: repeatedly move a point to a position on a line defined by two other points
    if it reduces the ordinary line count.
    """
    current_points = [p[:] for p in points]
    current_score = count_ordinary_lines(current_points)
    
    if is_all_collinear(current_points):
        return current_points, current_score

    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Pick a random point to move
        i = random.randint(0, 11)
        orig_point = current_points[i][:]
        
        # Pick two other points to define a line
        others = [j for j in range(12) if j != i]
        random.shuffle(others)
        a = others[0]
        b = others[1]
        
        # Try to find a new position for point i on line ab
        new_point = get_integer_point_on_line(current_points[a], current_points[b], t_range=(-5, 5))
        
        if new_point is None:
            continue
            
        # Check distinctness
        distinct = True
        for j in range(12):
            if j == i:
                continue
            if current_points[j] == new_point:
                distinct = False
                break
        
        if not distinct:
            continue
            
        # Try the move
        current_points[i] = new_point
        
        if is_all_collinear(current_points):
            current_points[i] = orig_point
            continue
            
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            improved = True
        else:
            # Revert if not better (greedy)
            current_points[i] = orig_point
            
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s safety margin for 150s limit
    
    # Initial configurations
    # 1. 3x4 Grid
    init_1 = [[x, y] for x in range(3) for y in range(4)]
    # 2. 4x3 Grid
    init_2 = [[x, y] for x in range(4) for y in range(3)]
    # 3. Random "Star"
    init_3 = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
    # Ensure distinctness for init_3
    while len(set(map(tuple, init_3))) < 12:
        init_3 = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
        
    candidates = [init_1, init_2, init_3]
    
    best_points = None
    best_score = float('inf')
    
    # Run ILS for each initial candidate
    for init in candidates:
        if time.time() > deadline - 10:
            break
            
        current_points, current_score = local_search(init, max_iters=2000)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        # Iterated Local Search: Perturb and re-search
        perturb_count = 0
        while time.time() < deadline - 5:
            # Perturb: move 2-3 random points to random integer coordinates
            perturbed_points = [p[:] for p in current_points]
            num_perturb = random.randint(2, 3)
            indices = random.sample(range(12), num_perturb)
            for idx in indices:
                perturbed_points[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
                
            # Ensure distinctness
            if len(set(map(tuple, perturbed_points))) < 12:
                continue
                
            # Local search on perturbed
            new_points, new_score = local_search(perturbed_points, max_iters=1000)
            
            # Accept if better, or with some probability (simulated annealing style)
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Simple acceptance: accept if much worse? No, just restart if stuck.
                # For simplicity, if no improvement in a few iterations, break inner loop
                perturb_count += 1
                if perturb_count > 20:
                    break
            
            perturb_count = 0 # Reset if improved? No, just keep counting failures
            
            # Actually, let's just run a fixed number of perturbations per init
            if perturb_count >= 10:
                break

    # Final validation
    if best_points is None:
        best_points = init_1
        
    if is_all_collinear(best_points):
        # Fallback fix
        best_points[0][1] += 1
        while len(set(map(tuple, best_points))) < 12:
            best_points[0][0] += 1
            
    return best_points
