"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def _score(points):
    """Compute the max multiplicity of a positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    
    # Use numpy for faster vectorized area calculation
    # points is a list of (x, y) tuples, convert to array
    pts = np.array(points)
    
    # Calculate areas for all triples
    # To be efficient, we can compute determinants for all i, j, k
    # But 30^3 = 27000 is small enough for a loop, but vectorization is better.
    # Let's stick to a loop for clarity and low overhead in this specific size, 
    # or use a more clever vectorized approach.
    
    # Actually, for N=30, a pure python loop might be slow if called thousands of times.
    # Let's use a semi-vectorized approach or just a tight loop.
    # 4060 triangles * 30 restarts * 1000 steps = 12 million area calcs.
    # Python loop: ~1 microsecond per calc? 12 seconds. Might be tight.
    # Let's optimize _score using NumPy broadcasting.
    
    # We can compute all pairwise differences? No, that's O(N^2).
    # Let's just use the loop but optimize it slightly.
    areas = []
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i+1, n):
            x2, y2 = pts[j]
            dx12 = x2 - x1
            dy12 = y2 - y1
            for k in range(j+1, n):
                x3, y3 = pts[k]
                det = abs(dx12 * (y3 - y1) - dy12 * (x3 - x1))
                if det > 0:
                    areas.append(det)
    
    if not areas:
        return 0
    
    # Find max frequency
    # Using Counter is fast
    from collections import Counter
    counts = Counter(areas)
    return max(counts.values())

def _random_points():
    """Generate 30 distinct random integer points in [-30, 30]^2."""
    points = set()
    while len(points) < 30:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        points.add((x, y))
    return list(points)

def _neighbor(points):
    """Generate a neighbor configuration by moving one point slightly."""
    points = [p[:] for p in points] # Deep copy list of tuples
    idx = random.randint(0, len(points) - 1)
    axis = random.choice([0, 1])
    step = random.randint(-3, 3)
    if step == 0:
        return points
    
    new_p = list(points[idx])
    new_p[axis] += step
    
    # Check bounds
    if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
        return None
        
    # Check distinctness
    new_tuple = tuple(new_p)
    if new_tuple == tuple(points[idx]):
        return points # No change
        
    # Check if it collides with another point
    for i, p in enumerate(points):
        if i != idx and tuple(p) == new_tuple:
            return None
            
    points[idx] = new_tuple
    return points

def construct():
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    best_score = 0
    best_points = None
    
    # Initial best from a simple grid to have a baseline
    base_points = [(x, y) for x in range(6) for y in range(5)]
    base_points = [(x-2, y-2) for x, y in base_points] # Center it roughly
    best_score = _score(base_points)
    best_points = base_points[:]
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        current_points = _random_points()
        current_score = _score(current_points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points[:]
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.95
        steps_per_restart = 1000
        
        for step in range(steps_per_restart):
            if time.time() - start_time > max_time:
                break
                
            neighbor_pts = _neighbor(current_points)
            if neighbor_pts is None:
                continue
                
            new_score = _score(neighbor_pts)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_points = neighbor_pts
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points[:]
            
            T *= cooling
            
            # If T is very small, we can break or restart
            if T < 0.01:
                break
                
    return best_points
# EVOLVE-BLOCK-END
