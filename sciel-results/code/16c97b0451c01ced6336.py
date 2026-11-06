import time
import random
import math
from itertools import combinations

def _collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return ('v', p1[0])
    if dy == 0:
        return ('h', p1[1])
    # Normalize slope and intercept
    # y = mx + c
    # To avoid floats, use integer representation
    # dy/dx = m
    # c = y1 - m*x1 = (y1*dx - x1*dy) / dx
    # We can represent the line by (dy, dx, y1*dx - x1*dy) normalized by GCD
    g = math.gcd(abs(dy), abs(dx))
    dy_n = dy // g
    dx_n = dx // g
    c_num = p1[1] * dx_n - p1[0] * dy_n
    # Ensure unique representation
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
        c_num = -c_num
    return (dy_n, dx_n, c_num)

def _analyze_points(points):
    """
    Returns:
    valid: bool (no 4 collinear)
    num_3_lines: int (number of lines with exactly 3 points)
    """
    lines = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    
    num_3_lines = 0
    for key, pts in lines.items():
        if len(pts) == 3:
            num_3_lines += 1
        elif len(pts) > 3:
            return False, 0
            
    return True, num_3_lines

def _random_point():
    return [random.randint(-100, 100), random.randint(-100, 100)]

def _generate_initial():
    points = []
    attempts = 0
    while len(points) < 20 and attempts < 10000:
        p = _random_point()
        if p not in points:
            points.append(p)
            valid, _ = _analyze_points(points)
            if not valid:
                points.pop()
            else:
                attempts = 0
        else:
            attempts += 1
    # If we couldn't generate 20, just return what we have (shouldn't happen)
    while len(points) < 20:
        p = _random_point()
        if p not in points:
             points.append(p)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial Configuration
    # Try a few random starts and pick the best one
    best_points = None
    best_score = -1
    
    for _ in range(10):
        pts = _generate_initial()
        valid, score = _analyze_points(pts)
        if valid and score > best_score:
            best_score = score
            best_points = pts[:]
            
    if best_points is None:
        # Fallback: Parabola (Score 20, but valid)
        return [[x, x * x] for x in range(-10, 10)]

    current_points = best_points[:]
    current_score = best_score
    best_points_sa = current_points[:]
    best_score_sa = current_score
    
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        new_points = current_points[:]
        idx = random.randint(0, 19)
        new_points[idx] = _random_point()
        
        # Check validity and score
        valid, new_score = _analyze_points(new_points)
        
        if not valid:
            # Reject invalid moves
            continue
            
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score_sa:
                best_score_sa = current_score
                best_points_sa = current_points[:]
        else:
            # Simulated Annealing acceptance
            delta = new_score - current_score
            # delta is <= 0
            if random.random() < math.exp(delta / T):
                current_points = new_points
                current_score = new_score
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_points_sa
