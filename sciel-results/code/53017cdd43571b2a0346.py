import time
from math import gcd

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            
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
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # The theoretical minimum for 12 points is 6 ordinary lines.
    # This is achieved by the affine image of the Pappus configuration extended,
    # or more simply, by specific configurations like the "Hesse" or "Pappus" type.
    
    # A well-known configuration for 12 points with 6 ordinary lines is based on 
    # a 3x4 grid that has been "broken" to create more collinearities.
    # However, a standard 3x4 grid has more ordinary lines.
    
    # Let's try to construct the optimal configuration.
    # One optimal configuration for n=12 with 6 ordinary lines is derived from the 
    # Pappus configuration (9 points, 9 lines, 4 points per line) plus 3 more points.
    # But a simpler, known optimal configuration for 12 points is:
    # Points on 3 parallel lines, with specific offsets.
    
    # Actually, the minimum number of ordinary lines for 12 points is 6.
    # A configuration achieving this is:
    # 3 lines, each containing 4 points. The lines intersect in 3 points (forming a triangle).
    # Wait, 3 lines of 4 points: 
    # If they form a triangle, the 3 intersection points are shared.
    # Total points = 3*4 - 3 = 9. We need 12 points.
    
    # Let's look for a known configuration.
    # The "near-pencil" is bad.
    # The grid is decent.
    
    # A specific configuration that achieves 6 ordinary lines for 12 points:
    # It is based on the complete quadrilateral or similar structures.
    
    # Let's try a deterministic search over a small set of candidate structures.
    # Structure 1: 3x4 Grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    
    # Structure 2: A "staircase" or perturbed grid.
    # Let's try to find a configuration where many lines have 3 or 4 points.
    
    # A known optimal configuration for 12 points:
    # Points: 
    # (0,0), (1,0), (2,0), (3,0)  -> 4 points on y=0
    # (0,1), (1,1), (2,1), (3,1)  -> 4 points on y=1
    # (0,2), (1,2), (2,2), (3,2)  -> 4 points on y=2
    # This is the 4x3 grid.
    
    # Let's count ordinary lines for the 4x3 grid.
    # Horizontal lines: 3 lines with 4 points each. (0 ordinary)
    # Vertical lines: 4 lines with 3 points each. (0 ordinary)
    # Diagonals:
    # Slope 1: (0,0)-(1,1)-(2,2) -> 3 points. (1,1)-(2,2)-(3,3) -> (3,3) is not in grid.
    # Wait, grid is x in 0..3, y in 0..2.
    # (0,0)-(1,1)-(2,2) -> 3 points.
    # (1,0)-(2,1)-(3,2) -> 3 points.
    # (0,1)-(1,2) -> 2 points.
    # (0,0)-(1,0) is vertical.
    
    # Slope -1:
    # (0,2)-(1,1)-(2,0) -> 3 points.
    # (1,2)-(2,1)-(3,0) -> 3 points.
    # (0,1)-(1,0) -> 2 points.
    
    # Slope 2:
    # (0,0)-(1,2) -> 2 points.
    # (1,0)-(2,2) -> 2 points.
    
    # Slope 1/2:
    # (0,0)-(2,1) -> 2 points.
    # (1,0)-(3,1) -> 2 points.
    
    # It seems the grid still has many ordinary lines.
    
    # Let's try a different approach: Simulated Annealing or Random Restart Hill Climbing.
    # But we need a good seed.
    
    # A very good seed is the "Pappus" configuration.
    # Pappus: 9 points.
    # Add 3 points.
    
    # Let's try a specific known optimal configuration for 12 points.
    # Reference: "Ordinary lines" by Guth and Katz, or earlier work by Green and Tao.
    # For n=12, the minimum is 6.
    # One such configuration is:
    # 3 lines of 4 points, forming a triangle, plus 3 points at infinity? No, we need affine.
    
    # Let's try to construct it:
    # Line 1: (0,0), (1,0), (2,0), (3,0)
    # Line 2: (0,1), (2,1), (4,1), (6,1)  -> Slope 0, y=1
    # Line 3: (0,2), (3,2), (6,2), (9,2)  -> Slope 0, y=2
    # This is just 3 parallel lines.
    
    # Let's try intersecting lines.
    # Line 1: (0,0), (1,1), (2,2), (3,3)
    # Line 2: (0,1), (1,0), (2,-1), (3,-2)
    # Line 3: (0,2), (1,3), (2,4), (3,5)
    # These are 3 parallel lines (slope 1, -1, 1). Not good.
    
    # Let's use a random search with a strong local optimization step.
    # We will start with the 3x4 grid and try to reduce ordinary lines.
    
    best_points = [p[:] for p in grid_3x4]
    best_count = count_ordinary_lines(best_points)
    
    # If we already have 6, we are done.
    if best_count <= 6:
        return best_points
    
    # Local search:
    # For each point, try moving it to a nearby integer coordinate.
    # To speed up, we can use a delta evaluation, but for simplicity and robustness,
    # we will use the full count but limit the search space.
    
    # However, a full count is O(n^2) = 144 operations, which is very fast.
    # So we can do many iterations.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_x, old_y = best_points[i]
            
            # Try moves in a small neighborhood
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    # Check bounds
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check distinctness
                    new_point = (new_x, new_y)
                    is_distinct = True
                    for j in range(12):
                        if j != i and (best_points[j][0], best_points[j][1]) == new_point:
                            is_distinct = False
                            break
                    
                    if not is_distinct:
                        continue
                    
                    # Temporarily move
                    best_points[i] = [new_x, new_y]
                    new_count = count_ordinary_lines(best_points)
                    
                    if new_count < best_count:
                        best_count = new_count
                        improved = True
                    else:
                        best_points[i] = [old_x, old_y]
                        
            if time.time() > deadline:
                break
        
        # If we improved, we might want to try a larger jump or restart.
        # For now, we just continue.
        
    # If we are stuck, try a random restart.
    if best_count > 6 and time.time() < deadline:
        import random
        random.seed(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Generate a random configuration
            random_points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
            
            # Ensure distinctness
            unique = set()
            valid = True
            for p in random_points:
                if (p[0], p[1]) in unique:
                    valid = False
                    break
                unique.add((p[0], p[1]))
            
            if not valid:
                continue
            
            random_count = count_ordinary_lines(random_points)
            
            if random_count < best_count:
                best_points = [p[:] for p in random_points]
                best_count = random_count
                
                # Local search from here
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for i in range(12):
                        old_x, old_y = best_points[i]
                        for dx in range(-5, 6):
                            for dy in range(-5, 6):
                                if dx == 0 and dy == 0:
                                    continue
                                new_x = old_x + dx
                                new_y = old_y + dy
                                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                                    continue
                                new_point = (new_x, new_y)
                                is_distinct = True
                                for j in range(12):
                                    if j != i and (best_points[j][0], best_points[j][1]) == new_point:
                                        is_distinct = False
                                        break
                                if not is_distinct:
                                    continue
                                best_points[i] = [new_x, new_y]
                                new_count = count_ordinary_lines(best_points)
                                if new_count < best_count:
                                    best_count = new_count
                                    improved = True
                                else:
                                    best_points[i] = [old_x, old_y]
                                if time.time() > deadline:
                                    break

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print(count_ordinary_lines(pts))
