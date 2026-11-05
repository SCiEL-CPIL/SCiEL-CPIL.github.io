import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial configuration: 3x4 grid
    # Points: (0,0), (1,0), (2,0), (3,0)
    #          (0,1), (1,1), (2,1), (3,1)
    #          (0,2), (1,2), (2,2), (3,2)
    points = [[x, y] for x in range(4) for y in range(3)]
    
    def count_ordinary_lines(pts):
        """
        Count the number of lines containing exactly two points.
        """
        if len(pts) < 2:
            return 0
            
        # Map from line key (dx, dy normalized) + intercept to list of points
        # Using a dictionary: key -> set of point indices
        # A line is defined by two points (i, j).
        # To avoid duplicates, we can iterate over all pairs and group them.
        
        # Efficient approach:
        # For each pair (i, j), calculate the line.
        # If a line has k points, it contributes to the count of "lines with >= 2 points".
        # We want the number of lines with exactly 2 points.
        
        # Let's collect all lines that have at least 2 points.
        # A line is uniquely identified by its slope and intercept.
        # For integer points, we can represent the line by the normalized vector (dx, dy) 
        # and a constant C such that dy*x - dx*y = C.
        # (dx, dy) should be coprime and dy >= 0 (or some canonical form).
        
        lines = {}
        
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                
                # Normalize direction vector
                g = gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g
                    dy //= g
                
                # Canonical form for direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                    
                # Calculate constant C = dy*x - dx*y
                # Using x1, y1
                c = dy * x1 - dx * y1
                
                key = (dx, dy, c)
                
                if key not in lines:
                    lines[key] = set()
                
                lines[key].add(i)
                lines[key].add(j)
        
        ordinary_count = 0
        for key, point_indices in lines.items():
            if len(point_indices) == 2:
                ordinary_count += 1
                
        return ordinary_count

    current_points = [p[:] for p in points]
    current_score = count_ordinary_lines(current_points)
    
    # Local Search
    # We will try to move each point to a nearby location.
    # To be efficient, we define a "neighborhood" of candidate positions for each point.
    # Since the grid is small, we can search in a box around the current position.
    
    # To avoid infinite loops on plateaus, we can use a simple first-improvement strategy
    # or accept equal scores with some probability to escape local minima.
    
    no_improvement_count = 0
    max_no_improvement = 2000
    
    while time.time() < deadline:
        improved = False
        
        # Shuffle points to avoid bias towards specific indices
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            x, y = current_points[i]
            
            # Try moving to nearby integer coordinates
            # Range of search: -5 to 5 relative to current position
            # This might be too small or too large. 
            # Given the grid is 4x3, moving within the bounding box of the current cluster is good.
            # Let's try a box of size 10x10 around the current point, but restricted to avoid duplicates.
            
            best_new_score = current_score
            best_new_point = None
            
            # Candidate positions: 
            # 1. Nearby grid points
            # 2. Points that might complete lines
            
            # Simple strategy: try all points in a box [x-5, x+5] x [y-5, y+5]
            # Filter out points that are already in the set (except the one being moved)
            
            # Optimization: Instead of trying all, we can just try a few random ones 
            # or a deterministic set. Deterministic is better for reproducibility.
            
            # Let's try a deterministic sweep of a small box
            for nx in range(x - 4, x + 5):
                for ny in range(y - 4, y + 5):
                    if nx == x and ny == y:
                        continue
                    
                    # Check if (nx, ny) is already occupied by another point
                    is_occupied = False
                    for k in range(12):
                        if k == i:
                            continue
                        if current_points[k][0] == nx and current_points[k][1] == ny:
                            is_occupied = True
                            break
                    
                    if is_occupied:
                        continue
                    
                    # Calculate new score
                    # Temporarily replace point i
                    old_point = current_points[i]
                    current_points[i] = [nx, ny]
                    
                    new_score = count_ordinary_lines(current_points)
                    
                    # Restore
                    current_points[i] = old_point
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_point = [nx, ny]
            
            if best_new_point is not None:
                current_points[i] = best_new_point
                current_score = best_new_score
                improved = True
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
        if not improved:
            # If no improvement in a full pass, try a random jump or restart
            # To escape local minima, we can randomly perturb one point to a far location
            # or just break if we've been stuck for a while.
            if no_improvement_count > max_no_improvement:
                # Try a random restart with a different initial grid or perturbation
                # For now, let's just break to save time, as 3x4 grid local search 
                # often finds the global optimum or near-optimum quickly.
                break
        else:
            no_improvement_count = 0
            
    # Final check: ensure points are distinct and within bounds
    # The local search preserves distinctness.
    # Bounds: The initial grid is 0..3, 0..2. Moves are +/- 4.
    # So x in [-4, 7], y in [-4, 6]. Well within 10^9.
    
    return current_points

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print("Ordinary lines:", count_ordinary_lines(pts))
