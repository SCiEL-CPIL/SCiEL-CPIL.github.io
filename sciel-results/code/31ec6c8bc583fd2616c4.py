"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def count_ordinary_lines_fast(points):
    """
    Count lines containing exactly 2 of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None
        
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        g = math.gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
            
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def point_on_line(p, line_key):
        if line_key is None:
            return False
        dx, dy, c = line_key
        x, y = p
        return dy * x - dx * y == c

    unique_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is not None:
                unique_lines.add(key)
                
    ordinary_count = 0
    for line_key in unique_lines:
        count = 0
        for p in points:
            if point_on_line(p, line_key):
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = time.time() + 140.0
    
    best_points = None
    best_count = float('inf')
    
    def update_best(points, count):
        nonlocal best_points, best_count
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in points]
            
    # Baseline: 3x4 grid
    grid_points = [[x, y] for x in range(4) for y in range(3)]
    grid_count = count_ordinary_lines_fast(grid_points)
    update_best(grid_points, grid_count)
    
    # Strategy: Combinatorial search over line structures.
    # We define a set of "rich" lines (lines with >= 3 points).
    # The points are the union of these lines.
    # We search for configurations of lines such that the resulting point set has few ordinary lines.
    
    # For 12 points, a good configuration might involve:
    # - 4 lines of 3 points? (Total points could be 12 if disjoint, but intersections reduce count)
    # - 3 lines of 4 points?
    # - A mix.
    
    # Let's try to construct specific known configurations or search small integer coordinates
    # for points that lie on many lines.
    
    # Alternative: Local search but in the space of "permutations" or "assignments" to a grid?
    # No, let's try a different structural search:
    # Pick 3 lines. Place points on them.
    
    # Let's try a deterministic construction based on projective planes or similar, but scaled to integers.
    # Since we need integer points, we can use affine transformations of small integer configs.
    
    # Let's try to find a config with 6 ordinary lines by searching small integer boxes more intelligently.
    # Instead of random points, we can fix some points and solve for others to be collinear with existing points.
    
    # Example:
    # Fix (0,0), (1,0), (0,1).
    # Try to add points that create collinearities.
    
    # Let's use a "greedy" construction:
    # 1. Start with empty set.
    # 2. Add a point.
    # 3. Add a point such that it forms a line with previous points?
    
    # Actually, let's try a specific known config: the "Hessian" configuration has 9 points and 12 lines of 3.
    # We need 12 points.
    
    # Let's try a 4x3 grid but with some points moved to create more collinearities.
    # The 4x3 grid has 12 points.
    # Ordinary lines in 4x3 grid:
    # Total pairs: 66.
    # Lines with 3 points:
    #   Rows: 3 lines of 4 points. (3 lines)
    #   Cols: 4 lines of 3 points. (4 lines)
    #   Diagonals (slope 1): (0,0)-(1,1)-(2,2), (1,0)-(2,1)-(3,2). (2 lines)
    #   Diagonals (slope -1): (0,2)-(1,1)-(2,0), (1,2)-(2,1)-(3,0). (2 lines)
    #   Other slopes?
    #   Slope 1/2: (0,0)-(2,1)? No other. (0,1)-(2,2)? No other.
    #   Let's count carefully.
    
    # Let's just run a faster local search but with a different move operator:
    # Instead of moving a point by (dx,dy), try to "snap" a point to a line defined by two other points.
    
    def snap_search(points):
        pts = [p[:] for p in points]
        current_count = count_ordinary_lines_fast(pts)
        
        for _ in range(500): # Limit iterations
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Pick a random point i
            i = random.randint(0, 11)
            old_pt = pts[i]
            
            # Pick two other points j, k
            j, k = random.sample([x for x in range(12) if x != i], 2)
            
            # Define line through j and k
            xj, yj = pts[j]
            xk, yk = pts[k]
            
            # We want to move point i to a point on this line (or close to it)
            # such that it might create a new collinearity with a third point l.
            # Actually, just move i to a random point on the line defined by j and k,
            # within bounds, and check if it's distinct.
            
            # To make it efficient, we can pick a random integer point on the line segment or extension.
            # Or simply pick a random point in a box and check if it's collinear with j and k? No, that's rare.
            
            # Better: Move i to a point that is collinear with j and k AND with some other point l?
            # That's hard.
            
            # Let's try: Move i to a random point in a small box around its current position,
            # but with a bias towards lines defined by other points.
            
            # Actually, let's just do random restarts with a smarter initialization.
            # Initialize points on a few lines.
            
            # Break out of this inner loop for now, this function is a placeholder for a different strategy.
            return pts, current_count
            
    # Let's implement the "Line Structure" search.
    # We represent a configuration by a set of lines.
    # Each line is defined by a normal vector (a,b) and offset c, such that ax+by=c.
    # We want to choose a set of lines such that the intersection points (and points on lines) form a set of 12 points with few ordinary lines.
    
    # This is complex. Let's stick to a very aggressive local search with a different metric or move set.
    
    # Move set: Swap two points? No.
    # Move set: Move a point to a random integer coordinate in a larger box, but accept if it reduces ordinary lines.
    
    # Let's try to find a config with < 12 ordinary lines.
    # Current best is likely 12 (from 3x4 grid).
    
    # Let's try the "Pappus" configuration scaled up.
    # Pappus: 9 points, 9 lines of 3.
    # Can we add 3 points to Pappus to get 12 points and keep ordinary lines low?
    
    # Pappus points (in affine plane):
    # Lines L1, L2, L3 concurrent? No, Pappus uses two lines with 3 points each, and the intersections of cross-connections.
    
    # Let's try a specific 12-point config:
    # 4 lines of 3 points, arranged in a "grid" of lines?
    
    # Let's try a 2x6 grid?
    # Points: (0,0)..(5,0), (0,1)..(5,1)
    # Lines:
    # 2 rows of 6 points.
    # 6 cols of 2 points (ordinary).
    # Diagonals:
    # (0,0)-(1,1), (1,0)-(2,1)... all ordinary?
    # (0,0)-(2,1)? No.
    # This likely has many ordinary lines.
    
    # What about 3x4 grid again?
    # Let's calculate ordinary lines for 3x4 grid exactly.
    
    # Let's run a specialized search for the 3x4 grid variations.
    # We can permute the x-coordinates of the rows.
    # Row 0: x0, x1, x2, x3
    # Row 1: y0, y1, y2, y3
    # Row 2: z0, z1, z2, z3
    # We can assume y0=0, y1=1, y2=2 (affine transform).
    # We can assume x0=0, x1=1, x2=2, x3=3? No, we can shift/scale.
    # Let's fix Row 0: (0,0), (1,0), (2,0), (3,0).
    # Row 1: (a,1), (b,1), (c,1), (d,1).
    # Row 2: (e,2), (f,2), (g,2), (h,2).
    # We can fix a=0 by shifting x.
    # We can fix e=0 by shifting x? No, shift applies to all.
    # Let's fix Row 0: (0,0), (1,0), (2,0), (3,0).
    # Row 1: (0,1), (x1,1), (x2,1), (x3,1). (Fixed one point to remove x-translation)
    # Row 2: (0,2), (y1,2), (y2,2), (y3,2). (Fixed one point to remove x-translation)
    # We can also use y-scale to fix Row 1 y=1, Row 2 y=2.
    
    # So we have variables x1, x2, x3, y1, y2, y3.
    # We want to minimize ordinary lines.
    # We can search these 6 variables in a small range, e.g., -5 to 5.
    # 11^6 = 1.77 million possibilities. Very fast.
    
    random.seed(42)
    
    min_val = -10
    max_val = 10
    
    # We fix:
    # P0=(0,0), P1=(1,0), P2=(2,0), P3=(3,0)
    # P4=(0,1), P5=(x1,1), P6=(x2,1), P7=(x3,1)
    # P8=(0,2), P9=(y1,2), P10=(y2,2), P11=(y3,2)
    
    # We must ensure all points are distinct.
    # P4=(0,1) is distinct from Row 0.
    # P8=(0,2) is distinct from Row 0,1.
    # x1,x2,x3 must be distinct from 0 and each other.
    # y1,y2,y3 must be distinct from 0 and each other.
    # Also, points in different rows are distinct because y is different.
    
    for x1 in range(min_val, max_val+1):
        if time.time() > deadline - 10.0:
            break
        for x2 in range(min_val, max_val+1):
            if x2 == x1 or x2 == 0: continue
            for x3 in range(min_val, max_val+1):
                if x3 == x1 or x3 == x2 or x3 == 0: continue
                for y1 in range(min_val, max_val+1):
                    if y1 == 0: continue
                    for y2 in range(min_val, max_val+1):
                        if y2 == y1 or y2 == 0: continue
                        for y3 in range(min_val, max_val+1):
                            if y3 == y1 or y3 == y2 or y3 == 0: continue
                            
                            pts = [
                                [0,0], [1,0], [2,0], [3,0],
                                [0,1], [x1,1], [x2,1], [x3,1],
                                [0,2], [y1,2], [y2,2], [y3,2]
                            ]
                            
                            # Check distinctness (should be guaranteed by constraints)
                            if len(set(tuple(p) for p in pts)) < 12:
                                continue
                                
                            cnt = count_ordinary_lines_fast(pts)
                            update_best(pts, cnt)
                            
                            if best_count == 6:
                                return best_points

    # If the structured search didn't find 6, do a final local polish on the best found.
    if best_points:
        def final_polish(points):
            pts = [p[:] for p in points]
            current_count = count_ordinary_lines_fast(pts)
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                for i in range(12):
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            old_x, old_y = pts[i]
                            new_x, new_y = old_x + dx, old_y + dy
                            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                                continue
                            pts[i] = [new_x, new_y]
                            if len(set(tuple(p) for p in pts)) < 12:
                                pts[i] = [old_x, old_y]
                                continue
                            cnt = count_ordinary_lines_fast(pts)
                            if cnt < current_count:
                                current_count = cnt
                                improved = True
                            else:
                                pts[i] = [old_x, old_y]
            return pts, current_count
            
        improved_pts, improved_cnt = final_polish(best_points)
        update_best(improved_pts, improved_cnt)

    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

# EVOLVE-BLOCK-END
