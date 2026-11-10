"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import itertools
from collections import defaultdict

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map line (defined by normalized equation) to set of points on it
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue # Should not happen for distinct points
                
            dx_n = dx // g
            dy_n = dy // g
            
            # Normalize direction to ensure consistent key
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            # Line equation: dy_n * x - dx_n * y = c
            c = dy_n * x1 - dx_n * y1
            
            key = (dx_n, dy_n, c)
            
            if key not in line_points:
                line_points[key] = set()
            
            line_points[key].add((x1, y1))
            line_points[key].add((x2, y2))
            
    ordinary_count = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _generate_candidate_config():
    """
    Generates a candidate configuration of 12 points.
    We try a few structured approaches:
    1. Pappus-like or Grid-like structures.
    2. Random perturbations of small grids.
    3. Specific known low-ordinary-line configurations if we can find them.
    
    For 12 points, the theoretical minimum over reals is 6.
    A 3x4 grid has many ordinary lines.
    Let's try to construct a configuration with many collinearities.
    
    Strategy:
    - Start with a base structure (e.g., 3 lines of 4 points, or 4 lines of 3 points).
    - Ensure distinctness and integer coordinates.
    """
    # Approach 1: 3 parallel lines with 4 points each?
    # If we have 3 horizontal lines, each with 4 points, the lines themselves are not ordinary (4 points).
    # But vertical lines and diagonals might be ordinary.
    
    # Let's try a specific construction:
    # Points on x=0: (0,0), (0,1), (0,2), (0,3)
    # Points on x=1: (1,0), (1,1), (1,2), (1,3)
    # Points on x=2: (2,0), (2,1), (2,2), (2,3)
    # This is a 3x4 grid.
    
    # Let's try to "bend" the grid to create more collinearities.
    # Or try a different base.
    
    # A known good configuration for small N often involves points on a few lines.
    # Let's try to generate points on 3 lines, with 4 points on each line.
    # Line 1: y = 0, x in {0, 1, 2, 3} -> (0,0), (1,0), (2,0), (3,0)
    # Line 2: y = 1, x in {0, 1, 2, 3} -> (0,1), (1,1), (2,1), (3,1)
    # Line 3: y = 2, x in {0, 1, 2, 3} -> (0,2), (1,2), (2,2), (3,2)
    # This is the 3x4 grid again.
    
    # What if we use non-parallel lines?
    # Line 1: y=0, x in {0,1,2,3}
    # Line 2: y=x, points?
    # Let's just use a random search over a restricted space of "structured" points.
    
    # Let's try a configuration based on the "orchard problem" solutions.
    # For 12 points, a configuration with 6 ordinary lines might exist.
    
    # Let's try to build a set of 12 points with high symmetry.
    # e.g. 4 points on x-axis, 4 on y-axis, 4 on x=y?
    # (0,0) is shared.
    # Let's try:
    # P1: (0,0)
    # P2: (1,0), P3: (2,0), P4: (3,0)  -> Line y=0 has 4 pts.
    # P5: (0,1), P6: (0,2), P7: (0,3)  -> Line x=0 has 4 pts (incl origin).
    # P8: (1,1), P9: (2,2), P10: (3,3) -> Line y=x has 4 pts (incl origin).
    # We have 10 distinct points so far:
    # (0,0), (1,0), (2,0), (3,0), (0,1), (0,2), (0,3), (1,1), (2,2), (3,3)
    # We need 2 more points.
    # Let's add points to create more collinearities.
    # Maybe (1,2) and (2,1)?
    # Check line through (1,2) and (2,1): y = -x + 3.
    # Does it pass through any existing points?
    # (0,3) is on y=-x+3. So (0,3), (1,2), (2,1) are collinear.
    # Let's add (1,2) and (2,1).
    
    points = [
        [0, 0],
        [1, 0], [2, 0], [3, 0],
        [0, 1], [0, 2], [0, 3],
        [1, 1], [2, 2], [3, 3],
        [1, 2], [2, 1]
    ]
    
    # Check distinctness
    if len(set(map(tuple, points))) != 12:
        return None
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial configuration: The structured candidate above
    base_points = _generate_candidate_config()
    if base_points is None:
        # Fallback to grid
        base_points = [[x, y] for x in range(4) for y in range(3)]
        
    best_points = [p[:] for p in base_points]
    best_score = _count_ordinary_lines(best_points)
    
    # Local Search with Simulated Annealing
    # We will try to improve the score by moving points.
    
    # To speed up, we can precompute lines and counts, but for N=12, O(N^2) is fine.
    # 12*11/2 = 66 pairs.
    
    # Let's do a more aggressive local search.
    # We'll try moving each point to a set of candidate positions.
    
    # Candidate positions: 
    # 1. Nearby integer points.
    # 2. Points that would create new collinearities with existing pairs.
    
    # For now, let's stick to a robust hill-climbing with random restarts.
    
    num_restarts = 50
    max_iters_per_restart = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            current_points = [p[:] for p in best_points]
        else:
            # Random restart: perturb best points significantly
            current_points = []
            for p in best_points:
                # Move to a random point in a larger box, or keep some structure
                # Let's try mixing: keep some points, move others
                if random.random() < 0.5:
                    current_points.append(p[:])
                else:
                    # Random point in [-10, 10]
                    current_points.append([random.randint(-10, 10), random.randint(-10, 10)])
            
            # Ensure distinctness for restart
            # Simple check
            valid = True
            seen = set()
            for p in current_points:
                if tuple(p) in seen:
                    valid = False
                    break
                seen.add(tuple(p))
            if not valid:
                continue
                
            # Check not all collinear (unlikely with random)
            
        current_score = _count_ordinary_lines(current_points)
        
        for _ in range(max_iters_per_restart):
            if time.time() > deadline:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_x, old_y = current_points[idx]
            
            # Move to a random nearby integer coordinate
            # Increase range slightly to escape local minima
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
            
            # Check bounds
            if not (-10**9 <= new_x <= 10**9 and -10**9 <= new_y <= 10**9):
                continue
                
            # Check distinctness
            new_points = [p[:] for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            is_distinct = True
            for k, p in enumerate(new_points):
                if k == idx:
                    continue
                if p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
                    
            if not is_distinct:
                continue
            
            # Calculate score
            new_score = _count_ordinary_lines(new_points)
            
            # Accept if better
            if new_score < current_score:
                current_points = new_points
                current_score = new_score
            elif new_score == current_score:
                # Accept equal moves with some probability to shuffle
                if random.random() < 0.3:
                    current_points = new_points
            
        # Update best
        if current_score < best_score:
            best_points = [p[:] for p in current_points]
            best_score = current_score
            
    return best_points
# EVOLVE-BLOCK-END
