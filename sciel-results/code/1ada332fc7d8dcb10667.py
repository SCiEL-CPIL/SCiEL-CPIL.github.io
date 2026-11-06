import itertools
import random
import time

def construct():
    # Initial state: 4x4 grid
    # We want to maximize 4-point lines.
    # Grid has 10.
    
    def get_score(points):
        """
        Returns (is_valid, count_4_lines)
        is_valid is False if any 5 points are collinear.
        """
        n = len(points)
        # Map points to a set for quick lookup? 
        # We need to check collinearity.
        
        # To check for 5 collinear, we can check all 5-subsets? 
        # Or, find all lines with >= 4 points and check their size.
        
        # Efficient way:
        # For each pair (i, j), define a line.
        # Count how many points are on that line.
        # If count >= 5, invalid.
        # If count == 4, it's a 4-line.
        # But we must be careful not to double count the same line.
        
        lines_found = set()
        max_collinear = 0
        count_4 = 0
        
        # O(N^3) approach:
        # For each pair i, j, find the line.
        # For each k, check if on line.
        # This is 16*15*16/2 ~ 1920 operations. Very fast.
        
        for i in range(n):
            for j in range(i+1, n):
                p1 = points[i]
                p2 = points[j]
                # Line through p1, p2
                # dx, dy
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Check all other points
                # We need to identify the line uniquely.
                # A line can be represented by (A, B, C) for Ax + By + C = 0, normalized.
                # Or just use the set of points on the line.
                
                # Let's collect points on this line
                on_line = {i, j}
                for k in range(n):
                    if k == i or k == j:
                        continue
                    p3 = points[k]
                    # Cross product (p2-p1) x (p3-p1) == 0
                    if dx * (p3[1] - p1[1]) - dy * (p3[0] - p1[0]) == 0:
                        on_line.add(k)
                
                # Now we have the set of indices on this line
                if len(on_line) >= 5:
                    return False, 0 # Invalid
                
                if len(on_line) == 4:
                    # This is a 4-point line.
                    # We need to count it only once.
                    # The set of indices is a unique identifier for the line of points.
                    # But two different pairs (i,j) on the same line will generate the same set.
                    # So we can use a set of frozensets.
                    lines_found.add(frozenset(on_line))
        
        # After checking all pairs, lines_found contains all lines with exactly 4 points.
        # Wait, if a line has 5 points, we returned False.
        # If a line has 4 points, it will be found multiple times (once for each pair in the 4).
        # But we store the frozenset of indices, so duplicates are removed.
        # Is it possible that a line with 4 points is missed? No, any pair on it generates it.
        
        return True, len(lines_found)

    # Initial points
    points = [[x, y] for x in range(4) for y in range(4)]
    
    # Verify initial score
    valid, score = get_score(points)
    best_score = score
    best_points = [p[:] for p in points]
    
    # Search
    start_time = time.time()
    timeout = 110 # Safety margin
    
    # Local search
    # Try moving each point to a random location
    # Or better: try to snap points to lines formed by others?
    
    # Let's do a random restart hill climbing.
    # Since the space is large, pure random is bad.
    # Let's try to perturb the grid.
    
    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]
    
    # We will try to improve the grid.
    # If we can't, we try random restarts.
    
    # To make search effective, we should consider that lines are defined by slopes.
    # Common slopes: 0, inf, 1, -1, 1/2, -1/2, 2, -2, 1/3, -1/3, 3, -3.
    
    # Strategy:
    # 1. Start with grid.
    # 2. For each point, try moving it to a position that creates a new 4-line.
    #    How to find such positions?
    #    For any 3 other points, do they define a line? No, 3 points don't necessarily define a line unless collinear.
    #    If 3 points are collinear, the 4th point on that line is a candidate.
    #    If no 3 are collinear, we can't form a 4-line by just moving one point unless we move it to align with 3 others.
    #    So, check all triples of other points. If they are collinear, calculate the 4th point?
    #    Actually, if 3 points are collinear, any 4th point on that line forms a 4-line.
    #    But we only have 1 integer point to place.
    #    So, for each triple (i,j,k) of other points:
    #       If they are collinear, find the line.
    #       Check if there is an integer point on this line that is not one of them.
    #       If we place P_i there, we get a 4-line.
    #       But we must ensure we don't create a 5-line.
    
    # This is a specific heuristic. Let's combine with random search.
    
    def try_improve(points):
        current_valid, current_score = get_score(points)
        if not current_valid:
            return points, current_score
            
        improved = False
        for _ in range(1000): # Limit iterations per restart
            if time.time() - start_time > timeout:
                break
                
            # Pick a random point to move
            idx = random.randint(0, 15)
            orig_p = points[idx]
            
            # Generate candidate positions
            # 1. Random
            # 2. Positions on lines defined by 3 other points (if collinear)
            #    Actually, if 3 points are collinear, the line is determined.
            #    We can place the 4th point anywhere on that line.
            #    But we need integer coordinates.
            
            # Let's just try random moves first, and if stuck, try "snapping" to lines.
            
            # For now, simple random move
            new_p = random_point()
            # Ensure distinct
            if any(new_p == points[i] for i in range(16) if i != idx):
                continue
                
            points[idx] = new_p
            v, s = get_score(points)
            
            if v and s > current_score:
                current_score = s
                current_valid = v
                improved = True
            else:
                points[idx] = orig_p
                
            # If we improved, we might continue from here.
            # If not, we might need to accept worse moves (Simulated Annealing) or restart.
            # Given the small search space and clear objective, Hill Climbing with restarts is good.
            
            if improved:
                break # Restart the inner loop to try to improve further from new state
                
        return points, current_score

    # Main loop
    best_points, best_score = try_improve(points)
    
    # Try several restarts
    for restart in range(20):
        if time.time() - start_time > timeout:
            break
            
        # Perturb the best points found so far
        p = [pt[:] for pt in best_points]
        # Move 2-3 random points to random locations
        for _ in range(3):
            idx = random.randint(0, 15)
            p[idx] = random_point()
            # Ensure distinct
            while any(p[idx] == p[i] for i in range(16) if i != idx):
                p[idx] = random_point()
                
        p, s = try_improve(p)
        if s > best_score:
            best_score = s
            best_points = p
            
    return best_points

# Run once to see if it works
# Note: The function construct() is called by the evaluator.
