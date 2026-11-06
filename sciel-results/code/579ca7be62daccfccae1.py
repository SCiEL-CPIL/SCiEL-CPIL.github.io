import random
import numpy as np
from collections import Counter

random.seed(42)

def construct():
    """
    Construct 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area (absolute determinant).
    """
    
    def vectorized_max_multiplicity(points):
        """
        Calculate the maximum multiplicity of any single positive doubled area
        for a set of 30 points using NumPy vectorization.
        """
        pts = np.array(points)
        n = len(pts)
        
        # Generate all combinations of 3 indices
        # This creates a (num_combos, 3) array of indices
        # To avoid creating a huge list of indices in Python, we can use np.indices or similar,
        # but for N=30, num_combos is ~4060, which is small enough to generate directly.
        
        # Efficient way to get all triplets (i, j, k) with i < j < k
        # Using np.triu_indices is not directly applicable for 3 indices easily in one shot without loops,
        # but we can do it with a small loop or pre-computed indices.
        
        # Let's pre-compute the indices once outside the main loop if possible, 
        # but since points change, we just need the indices.
        
        # For N=30, C(30,3) = 4060.
        # We can create arrays for all triplets.
        
        # i, j, k
        # This is a bit heavy to do in pure numpy without loops for indices, 
        # but we can use a trick or just accept a small python loop for index generation 
        # which is constant cost.
        
        # Actually, let's just use itertools for indices and vectorize the math.
        # Or better, use the fact that we can compute all areas in a batch.
        
        # Let's try a different vectorization:
        # For each pair (i, j), the vector v = P[j] - P[i].
        # For each k, the area is 0.5 * |cross(v, P[k] - P[i])|.
        # This is still O(N^3) but with smaller constant factors if done in numpy blocks.
        
        # Let's stick to a optimized python loop for index generation but numpy for math.
        # Actually, for N=30, a pure python loop over 4060 triples is fast (~1ms).
        # The bottleneck in the original code was likely the overhead of `combinations` and object creation.
        
        # Let's use a simpler vectorized approach:
        # Create arrays of shape (4060, 3, 2) for the points? No, that's 4060*3*2 floats.
        
        # Let's just use the standard loop but optimized.
        areas = []
        pts_list = list(points)
        for i in range(n - 2):
            x1, y1 = pts_list[i]
            for j in range(i + 1, n - 1):
                x2, y2 = pts_list[j]
                for k in range(j + 1, n):
                    x3, y3 = pts_list[k]
                    # doubled area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                    da = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if da > 0:
                        areas.append(da)
        
        if not areas:
            return 0
        
        # Count frequencies
        # Using a dictionary or Counter is fast for ~4000 items
        counts = Counter(areas)
        return max(counts.values())

    # Simulated Annealing / Hill Climbing
    
    best_points = None
    best_score = 0
    
    # Initial configurations to try
    # 1. Random
    # 2. Grid
    # 3. "Star" or other structures
    
    initial_configs = []
    
    # Random
    for _ in range(5):
        pts = set()
        while len(pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            pts.add((x, y))
        initial_configs.append(list(pts))
        
    # Grid 6x5
    grid = [(x, y) for x in range(6) for y in range(5)]
    initial_configs.append(grid)
    
    # Grid 10x3
    grid2 = [(x, y) for x in range(10) for y in range(3)]
    initial_configs.append(grid2)
    
    # Grid 5x6
    grid3 = [(x, y) for x in range(5) for y in range(6)]
    initial_configs.append(grid3)
    
    # Try to find a good starting point
    for config in initial_configs:
        score = vectorized_max_multiplicity(config)
        if score > best_score:
            best_score = score
            best_points = config[:]
            
    # Annealing / Hill Climbing
    current_points = best_points[:]
    current_score = best_score
    
    # Parameters
    max_iterations = 5000
    temperature = 10.0
    cooling_rate = 0.999
    
    for i in range(max_iterations):
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        # Move to a nearby random point
        # Radius 5
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check uniqueness
        new_point = (new_x, new_y)
        if new_point in current_points:
            continue
            
        # Apply move
        current_points[idx] = new_point
        
        new_score = vectorized_max_multiplicity(current_points)
        
        # Acceptance criteria
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points[:]
        else:
            # Metropolis acceptance
            delta = new_score - current_score
            # If delta is negative, accept with probability exp(delta / temperature)
            # Since scores are integers, delta is integer.
            # We want to explore, so we use a simple probability
            if random.random() < np.exp(delta / temperature):
                # Accept
                current_score = new_score
            else:
                # Reject
                current_points[idx] = (old_x, old_y)
                
        temperature *= cooling_rate
        
    return best_points
