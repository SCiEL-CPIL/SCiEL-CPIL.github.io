import time
from math import gcd
from itertools import combinations

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the given points.
    Points are a list of [x, y] tuples or lists.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line key to count of points on that line
    # Line key: (dx, dy, c) where dx, dy are normalized direction vector
    # and c = dy*x - dx*y
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction
            if dx == 0 and dy == 0:
                continue # Should not happen for distinct points
            
            # Ensure unique representation
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
            # Calculate constant c = dy*x1 - dx*y1
            c = dy * x1 - dx * y1
            
            key = (dx, dy, c)
            
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
    
    ordinary_count = 0
    for key, count in line_counts.items():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    
    # The Pappus configuration is a (9_4) configuration: 9 points, 9 lines, 4 points on each line.
    # A 3x3 grid of points on a conic (like y=x^2) has 9 points.
    # The lines connecting them have specific intersection properties.
    # For 12 points, we can start with a 3x4 grid or a specific Pappus-based extension.
    
    # A known configuration for 12 points with few ordinary lines is derived from 
    # the Hesse configuration or similar finite geometry structures.
    
    # Let's try a specific deterministic construction:
    # 1. Start with a 3x4 grid of integer points.
    # 2. The 3x4 grid has 12 points.
    # 3. Let's count ordinary lines for a standard 3x4 grid.
    
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    cnt_3x4 = count_ordinary_lines(grid_3x4)
    
    # Let's try a 4x3 grid
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    cnt_4x3 = count_ordinary_lines(grid_4x3)
    
    # Let's try a 2x6 grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    cnt_2x6 = count_ordinary_lines(grid_2x6)
    
    # Let's try a 6x2 grid
    grid_6x2 = [[x, y] for x in range(2) for y in range(6)]
    cnt_6x2 = count_ordinary_lines(grid_6x2)
    
    # Let's try a 12x1 line (all collinear - invalid)
    
    # Let's try a specific Pappus-like configuration.
    # The Pappus configuration has 9 points.
    # We can add 3 more points to minimize ordinary lines.
    
    # However, a simpler approach is to use a known optimal or near-optimal configuration.
    # For n=12, the minimum number of ordinary lines is 6.
    # A configuration achieving 6 ordinary lines for 12 points is not trivially a grid.
    
    # Let's try to construct a configuration based on the Pappus configuration extended.
    # Pappus points: (i, i^2) for i in {-1, 0, 1} scaled?
    
    # Let's use a deterministic search over a small set of known good configurations.
    
    candidates = []
    
    # 1. 3x4 Grid
    candidates.append((cnt_3x4, grid_3x4))
    
    # 2. 4x3 Grid
    candidates.append((cnt_4x3, grid_4x3))
    
    # 3. 2x6 Grid
    candidates.append((cnt_2x6, grid_2x6))
    
    # 4. 6x2 Grid
    candidates.append((cnt_6x2, grid_6x2))
    
    # 5. A "Pappus-like" configuration
    # The Pappus configuration consists of 9 points.
    # Let's try to find a 12-point configuration by perturbing a grid or using a specific pattern.
    
    # Let's try a configuration with points on two lines, with some intersections.
    # For example, 6 points on x=0, 6 points on x=1.
    # This is the 2x6 grid.
    
    # Let's try a configuration with points on a parabola.
    # Points: (i, i^2) for i in range(-5, 7) -> 12 points.
    parabola = [[i, i*i] for i in range(-5, 7)]
    cnt_parabola = count_ordinary_lines(parabola)
    candidates.append((cnt_parabola, parabola))
    
    # Let's try a configuration with points on a circle?
    # Integer points on a circle are limited.
    
    # Let's try a configuration with points on a line and some off-line.
    # 11 points on x-axis, 1 point off.
    # This will have many ordinary lines.
    
    # Let's try to improve the best candidate found so far using a very fast local search.
    # But first, let's see if any of these are good.
    
    best_count = min(c[0] for c in candidates)
    best_points = [c[1] for c in candidates if c[0] == best_count][0]
    
    # If the best count is already 6, we are done.
    if best_count <= 6:
        return best_points
    
    # If not, we perform a very targeted local search.
    # We only allow moves that keep points within a small bounding box to ensure speed.
    # And we use a fast delta-evaluation if possible, but for simplicity and safety,
    # we will use the full count but limit the search space significantly.
    
    # However, the task says "Minimize the number of lines containing exactly two of the points (exact)."
    # And "Over the reals the minimum for 12 points is 6".
    
    # Let's check if the parabola configuration is good.
    # For points on a parabola, the number of ordinary lines is related to the number of secants.
    # For n points on a parabola, the number of ordinary lines is n*(n-1)/2 - (number of collinear triples).
    # Actually, for points on a conic, no 3 points are collinear unless they are on a line.
    # Wait, points on a parabola: any line intersects the parabola in at most 2 points.
    # So ALL lines determined by pairs of points on a parabola are ordinary lines!
    # So for 12 points on a parabola, the number of ordinary lines is 12*11/2 = 66.
    # That's very bad.
    
    # So the grid is likely better.
    # For a 3x4 grid, let's estimate.
    # Total lines: C(12,2) = 66.
    # Lines with 3 points:
    # Horizontal: 4 lines (each with 3 points) -> 4 lines.
    # Vertical: 3 lines (each with 4 points) -> 3 lines.
    # Diagonals:
    # Slope 1: (0,0)-(1,1)-(2,2) -> 1 line with 3 points.
    # (0,1)-(1,2)-(2,3) -> 1 line with 3 points.
    # (0,2)-(1,3) -> 2 points.
    # (1,0)-(2,1) -> 2 points.
    # Slope -1: (0,3)-(1,2)-(2,1) -> 1 line with 3 points.
    # (0,2)-(1,1)-(2,0) -> 1 line with 3 points.
    # Slope 1/2, 2, etc.
    
    # The grid likely has many ordinary lines.
    
    # Let's try a different configuration:
    # The "Borchardt" configuration or similar.
    
    # Since I cannot guarantee finding the 6-line configuration quickly without a more sophisticated search,
    # I will stick with the best of the simple candidates.
    # The 3x4 grid is a standard starting point.
    
    # Let's do a quick, bounded local search to see if we can improve the grid.
    # We will only try moves that are small and within a deadline.
    
    deadline = start_time + 140.0
    
    improved_points = [p[:] for p in best_points]
    current_count = best_count
    
    # Simple local search: try moving each point to a nearby integer coordinate.
    # To speed up, we will only do this for a few iterations.
    
    for _ in range(50): # 50 iterations
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
                
            old_x, old_y = improved_points[i]
            
            # Try a few nearby points
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    # Check if new point is distinct from others
                    new_point = (new_x, new_y)
                    is_distinct = True
                    for j in range(12):
                        if j != i and (improved_points[j][0], improved_points[j][1]) == new_point:
                            is_distinct = False
                            break
                            
                    if not is_distinct:
                        continue
                    
                    # Check bounds
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                        
                    # Temporarily move
                    improved_points[i] = [new_x, new_y]
                    new_count = count_ordinary_lines(improved_points)
                    
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                    else:
                        improved_points[i] = [old_x, old_y]
                        
        if not improved:
            break
            
    return improved_points
