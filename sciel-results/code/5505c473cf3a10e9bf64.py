import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    # Helper: Count collinear triples in a set of points
    def count_triples(points):
        count = 0
        n = len(points)
        # To optimize, we can just check all combinations of 3
        # 20 points -> 1140 combinations. Fast enough.
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    # Helper: Check validity (no 4 collinear)
    def is_valid(points):
        # If any 4 are collinear, invalid
        # We can check all combinations of 4? 20 choose 4 is 4845.
        # Or, for each pair, check how many other points are on that line.
        n = len(points)
        # Efficient check:
        # For each pair (i, j), count points k such that i, j, k are collinear.
        # If count >= 2 (i.e., 3 other points + i + j = 5 points? No, i,j + 2 others = 4 points total), 
        # wait. If i, j, k, m are collinear, then for pair (i,j), k and m are on the line.
        # So for any pair, the number of OTHER points on the line must be <= 1.
        # Total points on line <= 3.
        
        # Optimization: Use a set of lines? 
        # Simpler: Just check all 4-combinations if n is small, but 4845 is fine.
        # Actually, checking pairs is O(n^3) which is 8000 ops. 
        # Checking 4-combinations is O(n^4) which is 4845 * constant. 
        # Let's use the pair counting method.
        
        # Even faster: 
        # For each i, j:
        #   count = 0
        #   for k in others:
        #     if collinear(i, j, k): count += 1
        #   if count >= 2: return False (because i, j, k1, k2 are 4 collinear)
        
        for i in range(n):
            for j in range(i+1, n):
                count = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
                        if count >= 2:
                            return False
        return True

    # Helper: Generate a random valid point set
    def generate_random_valid():
        while True:
            points = []
            # Try to pick 20 distinct points from a grid
            # To avoid 4-collinear, we can pick from a "no 4 in a row" structure or just retry
            # A simple retry loop works if the space is large enough.
            # Grid: -100 to 100.
            # To make it faster, we can bias towards a grid that is less prone to 4-collinear?
            # Actually, random points in [-100, 100] rarely have 4 collinear.
            # Probability of 4 random points being collinear is low.
            # But we need 20.
            # Let's just try random distinct points.
            
            # To speed up, we can sample from a larger grid and check.
            # Or, use a structured grid with jitter?
            # Let's stick to random distinct points in [-50, 50] to keep numbers small.
            candidates = set()
            while len(candidates) < 20:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                candidates.add((x, y))
            points = list(candidates)
            if is_valid(points):
                return points

    # Initial generation
    current_points = generate_random_valid()
    current_cost = count_triples(current_points)
    best_points = current_points[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        # Move one point to a new random location
        idx = random.randint(0, 19)
        old_point = current_points[idx]
        
        # Try a few new locations for this point
        found_better = False
        for _ in range(10): # Try 10 random relocations per iteration
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            new_point = (new_x, new_y)
            
            # Check distinctness
            if new_point in current_points and new_point != old_point:
                continue
            
            # Create candidate
            candidate = current_points[:]
            candidate[idx] = new_point
            
            # Check validity
            if not is_valid(candidate):
                continue
            
            # Check cost
            cand_cost = count_triples(candidate)
            
            # Acceptance criterion
            if cand_cost < current_cost:
                # Always accept improvement
                current_points = candidate
                current_cost = cand_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_points = current_points[:]
                found_better = True
                break
            else:
                # Metropolis criterion
                delta = cand_cost - current_cost
                if T > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_points = candidate
                        current_cost = cand_cost
                        found_better = True
                        break
        
        if found_better:
            # If we found an improvement, we might want to cool down slower or reset?
            # Standard SA: cool down every step
            T *= cooling_rate
        else:
            T *= cooling_rate
            if T < 0.1:
                # Reheat slightly or reset?
                # Let's just keep going, T is small so only improvements accepted.
                # This becomes hill climbing.
                # To escape, we could restart from best_points with higher T.
                # But for simplicity, let's just let it run.
                pass

    # Final check: ensure best_points is valid
    # It should be, but just in case
    if not is_valid(best_points):
        # Fallback to a known valid set if something went wrong (unlikely)
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_points
