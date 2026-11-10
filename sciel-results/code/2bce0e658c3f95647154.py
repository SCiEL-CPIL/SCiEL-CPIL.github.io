"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict
from itertools import combinations

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue
                
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
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

def _count_ordinary_lines_fast(points):
    """
    Faster count using numpy-style vectorization where possible.
    For small n=12, the overhead of numpy may not help, so we use a 
    different approach: precompute all pairs and group by line.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Use a dictionary mapping line key to list of point indices
    line_to_indices = defaultdict(list)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue
                
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            c = dy_n * x1 - dx_n * y1
            
            key = (dx_n, dy_n, c)
            line_to_indices[key].append(i)
            line_to_indices[key].append(j)
    
    # For each line, count unique points
    ordinary_count = 0
    for key, indices in line_to_indices.items():
        unique_pts = set(indices)
        if len(unique_pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _known_good_configurations():
    """
    Return a list of known configurations that have low ordinary line counts.
    These are hand-crafted or from literature.
    """
    configs = []
    
    # Configuration 1: Based on the 4x3 grid with modifications
    # The 4x3 grid has 12 points. Let's analyze it.
    # Points: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)
    # This is a standard grid. Let's compute its ordinary lines.
    p1 = [[x, y] for y in range(3) for x in range(4)]
    configs.append(p1)
    
    # Configuration 2: A "bent" grid
    p2 = [[0,0], [1,0], [2,0], [3,0], 
          [0,1], [1,1], [2,1], [3,1],
          [0,2], [1,2], [2,2], [3,3]]
    configs.append(p2)
    
    # Configuration 3: Try to create more 3-point lines
    # Start with a 3x4 grid and perturb one point
    p3 = [[0,0], [1,0], [2,0], [3,0], 
          [0,1], [1,1], [2,1], [3,1],
          [0,2], [1,2], [2,2], [4,2]]
    configs.append(p3)
    
    # Configuration 4: Symmetric arrangement
    p4 = [[0,0], [2,0], [4,0], [1,1], [3,1], [5,1],
          [0,2], [2,2], [4,2], [1,3], [3,3], [5,3]]
    configs.append(p4)
    
    # Configuration 5: Try a hexagonal-like pattern
    p5 = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1],
          [0,2], [1,2], [2,2], [1,3], [2,3], [3,3]]
    configs.append(p5)
    
    # Configuration 6: Based on orchard-planting problem solutions
    # For 12 points, try to maximize 3-point lines
    p6 = [[0,0], [1,0], [2,0], [3,0],
          [0,1], [1,1], [2,1], [3,1],
          [0,2], [1,2], [2,2], [3,2]]
    # This is the same as p1, so let's try something different
    p6 = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1],
          [0,2], [1,2], [2,2], [3,0], [3,1], [3,2]]
    configs.append(p6)
    
    # Configuration 7: Try to create lines with 4 points
    p7 = [[0,0], [1,0], [2,0], [3,0],
          [0,1], [1,1], [2,1], [3,1],
          [0,2], [1,2], [2,2], [3,2]]
    # Again same as p1. Let's try a different structure.
    p7 = [[0,0], [1,0], [2,0], [3,0],
          [0,1], [1,1], [2,1], [3,1],
          [0,2], [1,2], [2,2], [4,1]]
    configs.append(p7)
    
    # Configuration 8: A more creative arrangement
    p8 = [[0,0], [2,0], [4,0], [6,0],
          [1,1], [3,1], [5,1], [7,1],
          [2,2], [4,2], [6,2], [8,2]]
    configs.append(p8)
    
    # Configuration 9: Try to use diagonal lines effectively
    p9 = [[0,0], [1,1], [2,2], [3,3],
          [0,1], [1,2], [2,3], [3,4],
          [0,2], [1,3], [2,4], [3,5]]
    configs.append(p9)
    
    # Configuration 10: Mix of horizontal, vertical, and diagonal
    p10 = [[0,0], [1,0], [2,0], [3,0],
           [0,1], [1,1], [2,1], [3,1],
           [0,2], [1,2], [2,2], [3,3]]
    configs.append(p10)
    
    return configs

def _local_search(points, max_iterations=1000, time_limit=5.0):
    """
    Perform local search by moving one point at a time to minimize ordinary lines.
    """
    start_time = time.time()
    current_points = [p[:] for p in points]
    current_score = _count_ordinary_lines_fast(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(12):
            # Try several random nearby positions
            for _ in range(20):
                new_x = current_points[i][0] + random.randint(-3, 3)
                new_y = current_points[i][1] + random.randint(-3, 3)
                
                # Check if the new position is valid (distinct from other points)
                valid = True
                for j in range(12):
                    if j != i:
                        if current_points[j][0] == new_x and current_points[j][1] == new_y:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Create a new configuration
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                
                # Evaluate
                new_score = _count_ordinary_lines_fast(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                    
                    break  # Stop trying this point after improvement
        
        if not improved:
            # No improvement found, try a larger jump
            i = random.randint(0, 11)
            new_x = current_points[i][0] + random.randint(-10, 10)
            new_y = current_points[i][1] + random.randint(-10, 10)
            
            valid = True
            for j in range(12):
                if j != i:
                    if current_points[j][0] == new_x and current_points[j][1] == new_y:
                        valid = False
                        break
            
            if valid:
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                new_score = _count_ordinary_lines_fast(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
    
    return best_points, best_score

def _simulated_annealing(points, time_limit=10.0):
    """
    Perform simulated annealing to find a better configuration.
    """
    start_time = time.time()
    current_points = [p[:] for p in points]
    current_score = _count_ordinary_lines_fast(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    temperature = 10.0
    cooling_rate = 0.95
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        i = random.randint(0, 11)
        new_x = current_points[i][0] + random.randint(-5, 5)
        new_y = current_points[i][1] + random.randint(-5, 5)
        
        # Check validity
        valid = True
        for j in range(12):
            if j != i:
                if current_points[j][0] == new_x and current_points[j][1] == new_y:
                    valid = False
                    break
        
        if not valid:
            continue
        
        new_points = [p[:] for p in current_points]
        new_points[i] = [new_x, new_y]
        new_score = _count_ordinary_lines_fast(new_points)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_points = new_points
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        temperature *= cooling_rate
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with known good configurations
    candidates = _known_good_configurations()
    
    # Evaluate all candidates
    scored_candidates = []
    for cand in candidates:
        score = _count_ordinary_lines_fast(cand)
        scored_candidates.append((score, cand))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_score, best_points = scored_candidates[0]
    
    # If we already have a good score, we can stop early
    if best_score <= 6:
        return best_points
    
    # Perform local search from the best candidate
    improved_points, improved_score = _local_search(best_points, max_iterations=2000, time_limit=30.0)
    if improved_score < best_score:
        best_score = improved_score
        best_points = improved_points
    
    if best_score <= 6:
        return best_points
    
    # Try simulated annealing from the best point
    if time.time() < deadline - 20:
        sa_points, sa_score = _simulated_annealing(best_points, time_limit=20.0)
        if sa_score < best_score:
            best_score = sa_score
            best_points = sa_points
    
    if best_score <= 6:
        return best_points
    
    # Try local search from other good candidates
    for score, cand in scored_candidates[1:5]:
        if time.time() > deadline - 10:
            break
        
        ls_points, ls_score = _local_search(cand, max_iterations=500, time_limit=5.0)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
        
        if best_score <= 6:
            break
    
    # Final local search from the best
    if time.time() < deadline - 5:
        final_points, final_score = _local_search(best_points, max_iterations=1000, time_limit=5.0)
        if final_score < best_score:
            best_score = final_score
            best_points = final_points
    
    return best_points
# EVOLVE-BLOCK-END
