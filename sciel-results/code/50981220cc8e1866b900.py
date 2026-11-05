import time
import random
import math
from math import gcd

def count_ordinary_lines(points):
    """
    Counts the number of lines that contain exactly 2 points from the given set.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    # Helper to get a canonical key for the line passing through p1 and p2
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # Line equation: A*x + B*y + C = 0
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize the coefficients
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        
        # Ensure a unique representation
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)
        
    # Iterate over all pairs
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    # Count lines with exactly 2 points
    ordinary_count = sum(1 for v in lines.values() if v == 2)
    return ordinary_count

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    # Initial Configuration
    # A known configuration with few ordinary lines is often based on grids or specific symmetric shapes.
    # Let's start with a 3x3 grid (9 points) + 3 extra points.
    # The 3x3 grid has 0 ordinary lines? No.
    # Let's check 3x3 grid:
    # Points: (0,0)..(2,2)
    # Horizontal lines: y=0, y=1, y=2 (3 lines, 3 pts each)
    # Vertical lines: x=0, x=1, x=2 (3 lines, 3 pts each)
    # Diagonal lines: x+y=2 (3 pts), x-y=0 (3 pts), x+y=0 (1 pt), x+y=4 (1 pt), x-y=-2 (1 pt), x-y=2 (1 pt)
    # Wait, the grid points are (0,0) to (2,2).
    # Lines with 2 points:
    # (0,0)-(2,1): y=0.5x. Points: (0,0), (2,1). Only 2.
    # (0,1)-(2,0): y=-0.5x+1. Points: (0,1), (2,0). Only 2.
    # (0,2)-(2,1): y=-0.5x+2. Points: (0,2), (2,1). Only 2.
    # (1,0)-(2,2): y=2x-2. Points: (1,0), (2,2). Only 2.
    # (1,1)-(2,0): y=-x+2. Points: (1,1), (2,0). Only 2.
    # (0,0)-(1,2): y=2x. Points: (0,0), (1,2). Only 2.
    # (1,0)-(0,2): y=-2x+2. Points: (1,0), (0,2). Only 2.
    # (2,0)-(1,2): y=-2x+4. Points: (2,0), (1,2). Only 2.
    # (0,1)-(1,0) is same as (1,1)-(2,0)? No. (0,1)-(1,0) is y=-x+1.
    # Let's just use a random start or a simple grid and let SA do the work.
    
    # A better start: The "Hesse" configuration is 9 points, 12 lines of 3.
    # We need 12 points.
    # Let's try a 4x3 grid or similar.
    
    def random_config():
        pts = set()
        while len(pts) < 12:
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            pts.add((x, y))
        return [list(p) for p in pts]
    
    # Start with a structured config: 3x3 grid + 3 points far away to break symmetry?
    # Or just use the 3x3 grid and add 3 points on the "ordinary" lines to convert them to 3-point lines.
    
    # Let's start with a 3x3 grid centered at 0.
    base_pts = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]]
    # Add 3 points.
    # Let's try adding (2,0), (0,2), (-2,0)
    # This might create more 3-point lines.
    
    current_pts = base_pts + [(2, 0), (0, 2), (-2, 0)]
    current_score = count_ordinary_lines(current_pts)
    
    best_pts = [list(p) for p in current_pts]
    best_score = current_score
    
    # Candidate Pool Generation
    # We will generate candidates dynamically based on the current configuration.
    # 1. Nearby points (small perturbations)
    # 2. Points extending lines formed by pairs of current points.
    
    def generate_candidates(points):
        cands = set()
        # 1. Local perturbations
        for p in points:
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0: continue
                    cands.add((p[0] + dx, p[1] + dy))
        
        # 2. Line extensions
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Extend in both directions
                for k in [1, 2, 3]:
                    # Forward
                    cands.add((x2 + k * dx, y2 + k * dy))
                    # Backward
                    cands.add((x1 - k * dx, y1 - k * dy))
        
        # Filter out points already in the set and points that are too large
        current_set = set(map(tuple, points))
        valid_cands = []
        for c in cands:
            if c in current_set:
                continue
            if abs(c[0]) <= 10**9 and abs(c[1]) <= 10**9:
                valid_cands.append(c)
        
        return valid_cands

    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    max_steps = 20000  # Adjust based on time
    
    # Pre-calculate some static candidates if needed, but dynamic is better.
    # To speed up, we can limit the candidate pool size.
    
    step = 0
    while time.time() - start_time < timeout:
        if step % 100 == 0 and T < T_min:
            # Restart with a new random configuration if stuck
            # Or just break
            break
            
        # Generate candidate pool
        # To avoid overhead, we might reuse a pool for a few steps or limit its size
        cands = generate_candidates(current_pts)
        
        if not cands:
            T *= cooling
            step += 1
            continue
            
        # Pick a random candidate and a random index to replace
        idx = random.randint(0, 11)
        new_pt = random.choice(cands)
        
        # Ensure distinctness
        if new_pt == current_pts[idx]:
            continue
            
        new_pts = current_pts.copy()
        new_pts[idx] = new_pt
        
        # Check distinctness
        if len(set(map(tuple, new_pts))) != 12:
            continue
            
        new_score = count_ordinary_lines(new_pts)
        
        # Acceptance Probability
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_pts = new_pts
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_pts = [list(p) for p in current_pts]
                if best_score <= 6:
                    # Theoretical minimum is 6. If we hit it, we can stop or continue to ensure validity.
                    # But 6 is the lower bound.
                    break
                
        T *= cooling
        step += 1
        
    # Final check
    if count_ordinary_lines(best_pts) != best_score:
        # Recalculate just in case
        best_score = count_ordinary_lines(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
