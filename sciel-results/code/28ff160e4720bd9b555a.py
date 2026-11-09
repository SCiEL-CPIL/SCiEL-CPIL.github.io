import time
import random
import math
from collections import defaultdict

def _count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the given points.
    Points are a list of [x, y] integer pairs.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = defaultdict(int)
    
    # Iterate over all pairs of points
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for Ax + By + C = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize the line equation
            # Find GCD of A, B, C
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Enforce a consistent sign convention
            # e.g., first non-zero among A, B, C should be positive
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            line_counts[(A, B, C)] += 1
            
    # Count lines with exactly 2 points
    ordinary_count = 0
    for count in line_counts.values():
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial configuration: A 3x4 grid is a good starting point, 
    # but random might be better to avoid symmetry traps. 
    # Let's try a mix: start with a grid, then randomize slightly.
    # Actually, pure random in a small box is often effective for SA.
    
    # Let's use a box size that allows for interesting structures.
    # Pappus configuration can be represented with small integers.
    # Example: Points on 3 lines intersecting 3 other lines.
    # Let's start with random points in [-10, 10]
    
    def random_point():
        return [random.randint(-10, 10), random.randint(-10, 10)]
    
    # Initialize 12 distinct points
    points = []
    seen = set()
    while len(points) < 12:
        p = random_point()
        if tuple(p) not in seen:
            seen.add(tuple(p))
            points.append(p)
            
    best_points = [p[:] for p in points]
    best_score = _count_ordinary_lines(best_points)
    
    # Simulated Annealing parameters
    T = 5.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() < deadline:
        # Generate a neighbor
        # Strategy: Move 1 or 2 points to new random positions
        # This helps escape local minima where moving a single point by 1 is not enough
        
        num_moves = random.choice([1, 1, 2]) # Bias towards single moves
        new_points = [p[:] for p in points]
        
        # Select distinct indices to move
        indices_to_move = random.sample(range(12), num_moves)
        
        valid_move = True
        for idx in indices_to_move:
            # Try a few random perturbations
            for _ in range(5):
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_x = new_points[idx][0] + dx
                new_y = new_points[idx][1] + dy
                
                # Keep within reasonable bounds to avoid huge numbers
                # The task allows up to 1e9, but small numbers are better for SA stability
                # If we go out of bounds, wrap or clamp? 
                # Let's clamp to [-50, 50] to keep numbers small and manageable
                # Or just let them grow? Small integers are likely optimal.
                # Let's restrict to [-20, 20] to focus search
                if -20 <= new_x <= 20 and -20 <= new_y <= 20:
                    new_points[idx] = [new_x, new_y]
                    break
            
        # Check for distinctness
        if len(set(map(tuple, new_points))) < 12:
            continue
            
        # Check if all collinear (should be rare, but good to check)
        # If all collinear, score is 0 ordinary lines? 
        # Wait, if all collinear, there is 1 line with 12 points. 
        # No lines with exactly 2 points. So score is 0.
        # But the task says "not all collinear". 
        # So we must reject all-collinear configurations.
        
        # Check collinearity
        all_collinear = True
        x0, y0 = new_points[0]
        x1, y1 = new_points[1]
        for i in range(2, 12):
            xi, yi = new_points[i]
            if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
                all_collinear = False
                break
                
        if all_collinear:
            continue
            
        new_score = _count_ordinary_lines(new_points)
        
        # Metropolis acceptance
        delta = new_score - best_score
        if delta <= 0:
            points = new_points
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
                if best_score == 6: # Theoretical minimum
                    return best_points
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                points = new_points
                
        # Cooling
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # Optionally restart with higher T if stuck? 
            # For now, just keep going or slightly reheat if no improvement for long
            # Simple approach: if T is min, keep it min.
            
    return best_points
