import random
import time
from collections import defaultdict

def _count_exact_4_lines(points):
    """
    Returns a tuple (num_lines_exactly_4, is_valid).
    is_valid is False if any line has >= 5 points or if points are not distinct.
    """
    n = len(points)
    if n != 16:
        return 0, False
    
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 16:
        return 0, False
    
    # Map lines to counts of points
    # A line is represented by (A, B, C) from Ax + By + C = 0
    # Normalized: gcd(A, B, C) = 1, A > 0 or (A == 0 and B > 0)
    
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            if A == 0 and B == 0:
                # Should not happen if points are distinct
                continue
                
            g = __import__('math').gcd(__import__('math').gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure canonical form
            if A < 0:
                A, B, C = -A, -B, -C
            elif A == 0 and B < 0:
                A, B, C = -A, -B, -C
            
            lines[(A, B, C)].add(i)
    
    count_4 = 0
    for line_key, pts_on_line in lines.items():
        k = len(pts_on_line)
        if k >= 5:
            return 0, False
        if k == 4:
            count_4 += 1
            
    return count_4, True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration: 4x4 grid
    # Shifted to center or just 0..3
    base_points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in base_points]
    best_score, _ = _count_exact_4_lines(best_points)
    
    # Search space bounds
    # The 4x4 grid is 0..3. Let's search in -10..10 to allow movement.
    # We can expand bounds if we find good scores, but let's start tight for speed.
    min_coord = -10
    max_coord = 10
    
    current_points = [p[:] for p in base_points]
    current_score, current_valid = _count_exact_4_lines(current_points)
    
    if not current_valid:
        current_score = -1000
        
    temp = 1.0
    cooling = 0.9999
    
    # Simulated Annealing
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random new position in the bounding box
        new_x = random.randint(min_coord, max_coord)
        new_y = random.randint(min_coord, max_coord)
        
        # If too close to current, maybe perturb slightly? 
        # Random jump in box is fine.
        
        if new_x == old_x and new_y == old_y:
            continue
            
        # Save old
        current_points[idx] = [new_x, new_y]
        
        new_score, new_valid = _count_exact_4_lines(current_points)
        
        if not new_valid:
            new_score = -1000
            
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / temp):
            # Accept
            current_points[idx] = [new_x, new_y]
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Reject
            current_points[idx] = [old_x, old_y]
            
        temp *= cooling
        
        # Periodic restart or expansion?
        # If stuck for a while, we could randomize more.
        # But SA usually handles this.
        
    return best_points
