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
            # Find GCD of A, B, C
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                # This shouldn't happen for distinct points
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
        # count is number of pairs on this line
        # k points form k*(k-1)/2 pairs
        # k^2 - k - 2*count = 0
        # k = (1 + sqrt(1 + 8*count)) / 2
        disc = 1 + 8 * count
        sqrt_disc = math.isqrt(disc)
        if sqrt_disc * sqrt_disc == disc:
            k = (1 + sqrt_disc) // 2
            if k == 2:
                ordinary_count += 1
                
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Initial configuration: 3x4 grid
    # Points: (x, y) for x in 0..3, y in 0..2
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Local Search
    # We try to move one point at a time to a nearby location
    # to reduce the number of ordinary lines.
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 11)
        orig_x, orig_y = current_points[idx]
        
        # Generate a random offset
        # Limit range to avoid huge jumps initially, but allow some exploration
        # Since bounds are 1e9, we can be generous, but local moves are better for optimization
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        
        new_x = orig_x + dx
        new_y = orig_y + dy
        
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check distinctness
        new_point = [new_x, new_y]
        is_distinct = True
        for i, p in enumerate(current_points):
            if i == idx:
                continue
            if p[0] == new_x and p[1] == new_y:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Create new configuration
        new_points = [p[:] for p in current_points]
        new_points[idx] = new_point
        
        # Calculate new score
        new_score = count_ordinary_lines(new_points)
        
        # Acceptance criterion
        if new_score < current_score:
            # Always accept improvement
            current_points = new_points
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                
                # If we hit the theoretical minimum, we can stop or continue to verify
                if best_score == 6:
                    # We found the minimum. We can return early or continue to ensure no better (impossible)
                    # But maybe we can find a more "stable" one? No, 6 is min.
                    # Let's return immediately to save time? 
                    # The task says "Minimize". 6 is the lower bound.
                    # Returning 6 is optimal.
                    return best_points
        elif new_score == current_score:
            # Accept with some probability to explore the plateau
            if random.random() < 0.5:
                current_points = new_points
                # Score is same, so best_score doesn't change unless we track distinct configs?
                # We only care about the minimum value.
        else:
            # Reject worsening moves (or accept with very low prob if stuck, but simple greedy is often enough for this size)
            pass
            
    return best_points
