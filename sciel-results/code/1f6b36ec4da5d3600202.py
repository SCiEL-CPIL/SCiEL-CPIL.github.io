import time
import random
from collections import defaultdict

def count_4_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    points: list of (x, y) tuples
    """
    n = len(points)
    if n < 4:
        return 0
    
    line_counts = defaultdict(int)
    
    # Map each pair to a line key
    # Line key: (dx, dy, c) where y = m x + c, normalized
    # To handle vertical lines and avoid floats, use (dx, dy, c)
    # where dx = x2 - x1, dy = y2 - y1
    # Normalize dx, dy by gcd
    # c = dx * y1 - dy * x1
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize dx, dy
            g = gcd(dx, dy)
            if g != 0:
                dx //= g
                dy //= g
            
            # Ensure consistent sign (e.g., dx > 0 or dx == 0 and dy > 0)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
                
            # Calculate c such that dy * x - dx * y = c is constant for the line
            # Actually, the line equation is dy * x - dx * y = c
            # Let's verify:
            # dy * x1 - dx * y1 = dy * x2 - dx * y2
            # dy(x1-x2) = dx(y1-y2)
            # dy(-dx) = dx(-dy) -> -dy*dx = -dx*dy. Correct.
            
            c = dy * x1 - dx * y1
            
            key = (dx, dy, c)
            line_counts[key] += 1
            
    # Count lines with exactly 4 points
    # Each line with k points is counted C(k, 2) times in line_counts
    # We want lines where k == 4.
    # C(4, 2) = 6.
    # So if line_counts[key] == 6, it's a 4-point line.
    
    count = 0
    for count_val in line_counts.values():
        if count_val == 6:
            count += 1
            
    return count

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_4_lines(best_points)
    
    # Local Search
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Helper to get neighbors or target points
    # We will try moving points to complete 3-point lines
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to complete 3-point lines
        # Find all 3-point lines
        # We can reuse the line counting logic to get counts and points on lines
        # But for simplicity, let's just do a greedy move search
        
        # Generate candidate moves
        # For each point, try moving to a nearby location
        # Or, specifically, try to move a point to a target that completes a line
        
        # To be efficient, let's just do random local moves first
        # Then targeted moves
        
        # Targeted moves:
        # Identify 3-point lines
        line_counts = defaultdict(int)
        line_points = defaultdict(list)
        
        for i in range(16):
            x1, y1 = current_points[i]
            for j in range(i + 1, 16):
                x2, y2 = current_points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(dx, dy)
                if g != 0:
                    dx //= g
                    dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                c = dy * x1 - dx * y1
                key = (dx, dy, c)
                line_counts[key] += 1
                line_points[key].append(i)
                
        # Find lines with exactly 3 points (count = 3)
        # C(3,2) = 3
        # If count == 3, it's a 3-point line
        candidates = []
        
        for key, cnt in line_counts.items():
            if cnt == 3:
                # This is a 3-point line
                # Find the 4th point
                # We have 3 points: p1, p2, p3
                # The 4th point p4 must be on the line
                # We can extrapolate
                pts = line_points[key]
                p1 = current_points[pts[0]]
                p2 = current_points[pts[1]]
                p3 = current_points[pts[2]]
                
                # Determine the step
                # Sort points by x (or y if vertical)
                # Since they are collinear, we can order them
                # Let's find the vector between adjacent points
                # This might be complex if they are not evenly spaced
                # But in a lattice, they are integer.
                
                # Simpler: Try all integer points on the line within bounds
                # Line: dy*x - dx*y = c
                # We can parametrize
                # But we can just check nearby points?
                # No, the 4th point might be far.
                # However, if we move a point TO the 4th position, we need to know where it is.
                
                # Let's just try to move one of the existing points to a new position
                # that creates a new 4-point line?
                # That's hard to guess.
                
                # Alternative: Random local search
                pass

        # Let's do a simple random local search
        # Pick a random point, move it to a random nearby integer
        # If score improves, keep it
        
        for _ in range(1000): # 1000 attempts per iteration
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 15)
            orig_x, orig_y = current_points[idx]
            
            # Random small move
            new_x = orig_x + random.randint(-5, 5)
            new_y = orig_y + random.randint(-5, 5)
            
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Check distinctness
            # Ensure (new_x, new_y) is not occupied by another point
            # Since we are moving one point, we just need to ensure it doesn't collide
            # But we are replacing current_points[idx]
            # We need to check if (new_x, new_y) is in current_points[k] for k != idx
            is_valid = True
            for k in range(16):
                if k != idx and current_points[k] == [new_x, new_y]:
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Try the move
            current_points[idx] = [new_x, new_y]
            new_score = count_4_lines(current_points)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                # Revert
                current_points[idx] = [orig_x, orig_y]
                
        if not improved:
            # Try a bigger move or restart
            # Let's try swapping two points
            # Or just break and restart with a small perturbation
            # For now, let's just break and do a random restart
            # Perturb the best solution slightly
            current_points = [p[:] for p in best_points]
            for _ in range(10):
                idx = random.randint(0, 15)
                current_points[idx][0] += random.randint(-10, 10)
                current_points[idx][1] += random.randint(-10, 10)
                # Check bounds
                if abs(current_points[idx][0]) > 1000 or abs(current_points[idx][1]) > 1000:
                    current_points[idx][0] = 0
                    current_points[idx][1] = 0
            current_score = count_4_lines(current_points)
            
    return best_points
