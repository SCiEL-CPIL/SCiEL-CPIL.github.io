import random
import itertools
import math
import numpy as np

def construct():
    """
    Find 16 distinct integer points with |x|, |y| <= 1000, no five collinear,
    maximizing the number of lines containing exactly four of the points.
    
    Uses simulated annealing to perturb a random set of points.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Helper functions
    def get_lines(points):
        """Get all lines (as normalized tuples) and their point counts."""
        from collections import defaultdict
        line_points = defaultdict(set)
        
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                # Normalize the direction
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                # Normalize the line representation
                # A line can be represented by (a, b, c) where ax + by + c = 0
                # with gcd(a,b,c)=1 and a>0 or (a==0 and b>0)
                a = dy_n
                b = -dx_n
                c = -(a * x1 + b * y1)
                g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                # Canonical form
                if a < 0:
                    a, b, c = -a, -b, -c
                elif a == 0 and b < 0:
                    a, b, c = -a, -b, -c
                line_key = (a, b, c)
                line_points[line_key].add(i)
        
        return line_points
    
    def count_four_lines(points):
        """Count lines with exactly 4 points."""
        line_points = get_lines(points)
        count = 0
        for key, pts in line_points.items():
            if len(pts) == 4:
                count += 1
        return count
    
    def has_five_collinear(points):
        """Check if any line has 5 or more points."""
        line_points = get_lines(points)
        for key, pts in line_points.items():
            if len(pts) >= 5:
                return True
        return False
    
    def are_valid(points):
        """Check all constraints: distinct points, bounds, no five collinear."""
        if len(set(tuple(p) for p in points)) != 16:
            return False
        for x, y in points:
            if abs(x) > 1000 or abs(y) > 1000:
                return False
        if has_five_collinear(points):
            return False
        return True
    
    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]
    
    def random_initial():
        """Generate a random valid initial configuration."""
        while True:
            points = []
            for _ in range(16):
                p = random_point()
                if tuple(p) not in [tuple(q) for q in points]:
                    points.append(p)
            if not has_five_collinear(points):
                return points
    
    # Start with a reasonable initial configuration
    # Try the 4x4 grid first as a baseline
    grid_points = [[x, y] for x in range(4) for y in range(4)]
    # Shift to center
    grid_points = [[x - 2, y - 2] for x, y in grid_points]
    
    current = grid_points
    current_score = count_four_lines(current)
    
    # Simulated annealing
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    max_iter = 50000
    
    for it in range(max_iter):
        if T < T_min:
            break
        
        # Perturb: move one random point to a new random location
        new_points = [p[:] for p in current]
        idx = random.randint(0, 15)
        new_p = random_point()
        new_points[idx] = new_p
        
        # Check validity
        if not are_valid(new_points):
            T *= cooling
            continue
        
        new_score = count_four_lines(new_points)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = [p[:] for p in current]
        
        T *= cooling
    
    # Return the best configuration found
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
