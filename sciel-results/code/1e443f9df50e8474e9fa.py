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

def _is_all_collinear(points):
    """
    Check if all points lie on a single line.
    """
    n = len(points)
    if n < 3:
        return False
    
    # Pick first two distinct points
    p0 = points[0]
    p1 = None
    for p in points[1:]:
        if p != p0:
            p1 = p
            break
            
    if p1 is None:
        return True # All points are the same (degenerate, but technically collinear)
        
    # Check if all other points are collinear with p0 and p1
    x0, y0 = p0
    x1, y1 = p1
    
    for p in points[2:]:
        x2, y2 = p
        # Cross product (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0) should be 0
        if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) != 0:
            return False
            
    return True

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

def _hexagon_hexagram_config():
    """
    Construct 12 points as a regular hexagon plus the 6 vertices of the 
    hexagram (star of David) formed by extending the hexagon's sides.
    
    A regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees
    has 3 main diagonals (connecting opposite vertices) that all pass through the center.
    The "dual" hexagram is formed by the intersections of the extended sides.
    
    For a unit hexagon with vertices (cos(k*pi/3), sin(k*pi/3)) for k=0..5:
    - The 6 main diagonals connect opposite vertices and pass through origin
    - The 6 side extensions create a larger hexagram
    
    Let's use a simpler approach: take a regular hexagon and scale it,
    then add 6 more points that create many 3-point collinearities.
    
    Key insight: In a regular hexagon H with vertices v0..v5, the lines
    connecting v_i to v_{i+2} (skipping one) create 6 "short diagonals" 
    that form a smaller hexagram. The intersections of these with the 
    main diagonals can give us additional points.
    
    For integer coordinates, use the hexagon:
    (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -- this is a hexagon
    
    Actually, let's use the standard hexagon:
    (2,0), (1,1), (-1,1), (-2,0), (-1,-1), (1,-1)
    
    The 3 long diagonals pass through origin.
    The 6 short diagonals (v_i to v_{i+2}) create the hexagram vertices.
    """
    # Regular hexagon vertices (scaled for integer coords)
    # Using: (2,0), (1,1), (-1,1), (-2,0), (-1,-1), (1,-1)
    hex_vertices = [(2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)]
    
    # The 6 "hexagram" points: these are the outer vertices of the Star of David
    # formed by extending the sides of the hexagon.
    # For a regular hexagon, the hexagram vertices are at distance 2x from center
    # along the same directions as the hexagon vertices.
    # So: (4,0), (2,2), (-2,2), (-4,0), (-2,-2), (2,-2)
    hexagram_vertices = [(4, 0), (2, 2), (-2, 2), (-4, 0), (-2, -2), (2, -2)]
    
    points = [list(p) for p in hex_vertices + hexagram_vertices]
    return points

def _hexagon_hexagram_variant(scale=1):
    """
    Generate hexagon+hexagram with different scaling to explore the space.
    """
    hex_vertices = [(2*scale, 0), (1*scale, 1*scale), (-1*scale, 1*scale), 
                    (-2*scale, 0), (-1*scale, -1*scale), (1*scale, -1*scale)]
    hexagram_vertices = [(4*scale, 0), (2*scale, 2*scale), (-2*scale, 2*scale), 
                         (-4*scale, 0), (-2*scale, -2*scale), (2*scale, -2*scale)]
    points = [list(p) for p in hex_vertices + hexagram_vertices]
    return points

def _known_good_configurations():
    """
    Return a list of known configurations that have low ordinary line counts.
    These are hand-crafted or from literature.
    """
    configs = []
    
    # Configuration 0: Hexagon + hexagram (primary candidate)
    configs.append(_hexagon_hexagram_config())
    
    # Configuration 0b: Scaled hexagon + hexagram variants
    for scale in [2, 3]:
        configs.append(_hexagon_hexagram_variant(scale))
    
    # Configuration 1: Based on the 4x3 grid with modifications
    p1 = [[x, y] for y in range(3) for x in range(4)]
    configs.append(p1)
    
    # Configuration 2: A "bent" grid
    p2 = [[0,0], [1,0], [2,0], [3,0], 
          [0,1], [1,1], [2,1], [3,1],
          [0,2], [1,2], [2,2], [3,3]]
    configs.append(p2)
    
    # Configuration 3: Try to create more 3-point lines
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
    p6 = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1],
          [0,2], [1,2], [2,2], [3,0], [3,1], [3,2]]
    configs.append(p6)
    
    # Configuration 7: Try to create lines with 4 points
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
    
    # Configuration 11: 11 points on a line + 1 off (known to give 11 ordinary lines)
    p11 = [[i, 0] for i in range(11)] + [[5, 1]]
    configs.append(p11)
    
    return configs

def _local_search(points, max_iterations=1000, time_limit=5.0):
    """
    Perform local search by moving one point at a time to minimize ordinary lines.
    Uses a wider search neighborhood and tries structured moves.
    Rejects configurations where all points are collinear.
    """
    start_time = time.time()
    current_points = [p[:] for p in points]
    if _is_all_collinear(current_points):
        return current_points, float('inf')
        
    current_score = _count_ordinary_lines_fast(current_points)
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    # Precompute set of occupied points for fast validity check
    occupied = set(tuple(p) for p in current_points)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(12):
            old_x, old_y = current_points[i]
            occupied.discard((old_x, old_y))
            
            # Try several random nearby positions with varying step sizes
            step_sizes = [1, 2, 3, 5, 8]
            for step in step_sizes:
                if time.time() - start_time > time_limit:
                    break
                for _ in range(10):
                    new_x = old_x + random.randint(-step, step)
                    new_y = old_y + random.randint(-step, step)
                    
                    # Check if the new position is valid (distinct from other points)
                    if (new_x, new_y) in occupied:
                        continue
                    
                    # Create a new configuration
                    new_points = [p[:] for p in current_points]
                    new_points[i] = [new_x, new_y]
                    
                    # Check collinearity constraint
                    if _is_all_collinear(new_points):
                        continue
                        
                    # Evaluate
                    new_score = _count_ordinary_lines_fast(new_points)
                    
                    if new_score < current_score:
                        # Accept improvement
                        occupied.add((new_x, new_y))
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        
                        break  # Stop trying this step size after improvement
                
                if improved:
                    break
            
            occupied.add((old_x, old_y))
            if improved:
                break
        
        if not improved:
            # No improvement found, try a larger jump
            i = random.randint(0, 11)
            old_x, old_y = current_points[i]
            occupied.discard((old_x, old_y))
            
            step = random.randint(10, 30)
            new_x = old_x + random.randint(-step, step)
            new_y = old_y + random.randint(-step, step)
            
            if (new_x, new_y) not in occupied:
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                
                if not _is_all_collinear(new_points):
                    new_score = _count_ordinary_lines_fast(new_points)
                    
                    if new_score < current_score:
                        occupied.add((new_x, new_y))
                        current_points = new_points
                        current_score = new_score
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
            
            occupied.add((old_x, old_y))
    
    return best_points, best_score

def _simulated_annealing(points, time_limit=10.0):
    """
    Perform simulated annealing to find a better configuration.
    Rejects configurations where all points are collinear.
    """
    start_time = time.time()
    current_points = [p[:] for p in points]
    if _is_all_collinear(current_points):
        return current_points, float('inf')
        
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
        
        # Check collinearity constraint
        if _is_all_collinear(new_points):
            continue
            
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
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Start with known good configurations
    candidates = _known_good_configurations()
    
    # Evaluate all candidates, filtering out collinear ones
    scored_candidates = []
    for cand in candidates:
        if _is_all_collinear(cand):
            continue
        score = _count_ordinary_lines_fast(cand)
        scored_candidates.append((score, cand))
        
    if not scored_candidates:
        # Fallback: A simple non-collinear config
        return [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2], [3, 0], [3, 1], [3, 2]]

    scored_candidates.sort(key=lambda x: x[0])
    
    best_score, best_points = scored_candidates[0]
    
    # If we already have a good score, we can stop early
    if best_score <= 6:
        return best_points
    
    # Perform local search from the best candidate (hexagon+hexagram first)
    remaining_time = deadline - time.time()
    ls_time = min(40.0, remaining_time * 0.4)
    improved_points, improved_score = _local_search(best_points, max_iterations=5000, time_limit=ls_time)
    if improved_score < best_score:
        best_score = improved_score
        best_points = improved_points
    
    if best_score <= 6:
        return best_points
    
    # Try simulated annealing from the best point
    remaining_time = deadline - time.time()
    if remaining_time > 25:
        sa_time = min(20.0, remaining_time * 0.4)
        sa_points, sa_score = _simulated_annealing(best_points, time_limit=sa_time)
        if sa_score < best_score:
            best_score = sa_score
            best_points = sa_points
    
    if best_score <= 6:
        return best_points
    
    # Try local search from other good candidates
    for score, cand in scored_candidates[1:6]:
        if time.time() > deadline - 15:
            break
        
        ls_time = min(8.0, (deadline - time.time()) * 0.3)
        ls_points, ls_score = _local_search(cand, max_iterations=2000, time_limit=ls_time)
        if ls_score < best_score:
            best_score = ls_score
            best_points = ls_points
        
        if best_score <= 6:
            break
    
    # Final local search from the best
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        final_time = min(8.0, remaining_time * 0.5)
        final_points, final_score = _local_search(best_points, max_iterations=3000, time_limit=final_time)
        if final_score < best_score:
            best_score = final_score
            best_points = final_points
            
    # Final safety check: ensure not all collinear
    if _is_all_collinear(best_points):
        # Return the best non-collinear candidate from the initial list
        for score, cand in scored_candidates:
            if not _is_all_collinear(cand):
                return cand
        # Absolute fallback
        return [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2], [3, 0], [3, 1], [3, 2]]

    return best_points
# EVOLVE-BLOCK-END
