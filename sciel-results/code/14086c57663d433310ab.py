"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    """
    Uses Simulated Annealing to find a configuration of 20 points with |x|, |y| <= 100
    that has no four collinear points and minimizes the largest subset with no three collinear.
    
    Strategy:
    1. Initialize with a structured grid-like pattern to ensure high initial collinearity.
    2. Define the objective as maximizing the number of collinear triples (proxy for minimizing GP subset).
    3. Use Simulated Annealing to navigate the search space, accepting worse moves with a temperature-dependent probability.
    4. Strictly enforce the "no four collinear" constraint.
    """
    random.seed(42)
    N = 20
    LIMIT = 100
    
    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])
        
    def check_no_4_collinear(points):
        """Returns True if no 4 points are collinear."""
        n = len(points)
        # For each pair, check how many other points are on the line
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 4:
                    return False
        return True

    def count_triples(points):
        """Count distinct collinear triples."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    # Initialization: 
    # Try to create a configuration with many 3-collinearities.
    # A 4x5 grid has many 4-collinearities (rows, cols). 
    # We need to break the 4-collinearities while keeping 3s.
    # Let's start with random points but bias towards a grid.
    
    def generate_initial():
        # Try a few random restarts to find a valid start with high triples
        best_pts = None
        best_triples = -1
        for _ in range(50):
            pts = []
            used = set()
            for _ in range(N):
                while True:
                    x = random.randint(-LIMIT, LIMIT)
                    y = random.randint(-LIMIT, LIMIT)
                    if (x, y) not in used:
                        used.add((x, y))
                        break
                pts.append([x, y])
            if check_no_4_collinear(pts):
                t = count_triples(pts)
                if t > best_triples:
                    best_triples = t
                    best_pts = [p[:] for p in pts]
        if best_pts is None:
             # Fallback: purely random until valid (should happen quickly)
             while True:
                 pts = []
                 used = set()
                 for _ in range(N):
                     while True:
                         x = random.randint(-LIMIT, LIMIT)
                         y = random.randint(-LIMIT, LIMIT)
                         if (x, y) not in used:
                             used.add((x, y))
                             break
                     pts.append([x, y])
                 if check_no_4_collinear(pts):
                     return pts
        return best_pts

    points = generate_initial()
    current_triples = count_triples(points)
    best_points = [p[:] for p in points]
    best_triples = current_triples

    start_time = time.time()
    timeout = 110
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while T > T_min and time.time() - start_time < timeout:
        # Generate neighbor
        idx = random.randint(0, N - 1)
        old_point = points[idx]
        
        # Propose a new point
        # To stay within bounds and distinct, pick random
        new_point = [random.randint(-LIMIT, LIMIT), random.randint(-LIMIT, LIMIT)]
        
        if new_point == old_point:
            continue
        if any(p == new_point for p in points if p != old_point):
            continue
            
        points[idx] = new_point
        
        if not check_no_4_collinear(points):
            points[idx] = old_point
            continue
            
        new_triples = count_triples(points)
        delta = new_triples - current_triples
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta > 0:
            current_triples = new_triples
            if new_triples > best_triples:
                best_triples = new_triples
                best_points = [p[:] for p in points]
        else:
            # delta is negative or zero
            if delta == 0:
                # Accept equal to move around
                current_triples = new_triples
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_triples = new_triples
                else:
                    points[idx] = old_point
        
        T *= cooling_rate

    # Restore best found
    return best_points
# EVOLVE-BLOCK-END
