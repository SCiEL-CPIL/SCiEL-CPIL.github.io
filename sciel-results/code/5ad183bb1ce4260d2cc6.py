import random
import time
import math
from collections import defaultdict

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def line_rep(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle vertical/horizontal/normal
    if dx == 0 and dy == 0:
        return (0, 0, 0) # Should not happen
        
    # Normalize direction vector for unique line ID
    # We use the line equation Ax + By + C = 0
    A = dy
    B = -dx
    C = -(A * x1 + B * y1)
    
    # Normalize A, B, C
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
        
    A //= g
    B //= g
    C //= g
    
    # Ensure unique sign
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0:
        if B < 0:
            B, C = -B, -C
        elif B == 0 and C < 0:
            C = -C
            
    return (A, B, C)

def count_ordinary(points):
    counts = defaultdict(int)
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            rep = line_rep(pi, pj)
            counts[rep] += 1
    return sum(1 for v in counts.values() if v == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_points = None
    best_score = float('inf')
    
    # Initialize with a known decent config or random
    # Try a few random restarts
    for restart in range(50): # Max 50 restarts, break on time
        if time.time() > deadline:
            break
            
        # Random initialization
        # Use small range to encourage collisions
        points = [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(12)]
        
        # Ensure distinct points
        while len(set(points)) < 12:
            i = random.randint(0, 11)
            points[i] = (random.randint(-10, 10), random.randint(-10, 10))
        
        current_score = count_ordinary(points)
        if current_score < best_score:
            best_score = current_score
            best_points = points[:]
            
        # Local Search
        # Simulated Annealing style
        temp = 10.0
        for step in range(5000): # Fixed steps per restart, or time based
            if time.time() > deadline:
                break
                
            # Pick random point
            i = random.randint(0, 11)
            old_x, old_y = points[i]
            
            # Move to a nearby integer point
            # Delta x, delta y in [-2, 2]
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
                
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Optional: keep within bounds? 
            # The task allows |x|,|y| <= 10^9, so no strict bound needed for validity,
            # but small numbers help with integer overflow in line_rep (though Python handles big ints)
            # and keep the search space dense.
            
            points[i] = (new_x, new_y)
            
            # Check if distinct
            if points.count(points[i]) > 1:
                points[i] = (old_x, old_y)
                continue
                
            new_score = count_ordinary(points)
            
            delta = new_score - current_score
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / temp)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = points[:]
            else:
                points[i] = (old_x, old_y)
                
            temp = max(0.1, temp * 0.99)
            
        if best_score <= 6:
            break # Found theoretical minimum
            
    if best_points is None:
        # Fallback: return seed
        return [[x, y] for x in range(4) for y in range(3)]
        
    return [[x, y] for x, y in best_points]

# EVOLVE-BLOCK-END
