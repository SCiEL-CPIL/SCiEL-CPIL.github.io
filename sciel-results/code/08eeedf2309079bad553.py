import time
import math
import random

def construct():
    # Deadline guard: 150s wall-clock, use 140s for safety
    start_time = time.time()
    deadline = 140.0

    # Distance squared on triangular lattice
    # Points are a + b*omega, omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # (a1 + b1*omega) - (a2 + b2*omega) = (a1-a2) + (b1-b2)*omega
    # |z|^2 = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hex distance in (a,b) basis
    # Neighbors of (0,0) are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # hex_dist(a,b) = (abs(a) + abs(b) + abs(a+b)) // 2
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)

    # Generate initial hex spiral: take 32 points closest to origin by hex distance
    # Use a smaller search range for speed; hex_dist max needed is 3
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            candidates.append([a, b])
    
    candidates.sort(key=lambda p: (hex_dist(p), angle(p)))
    result = [p[:] for p in candidates[:32]]
    
    # Verify initial result
    best_edges = count_edges(result)
    target_edges = 76  # Harborth's bound for n=32
    
    if best_edges >= target_edges:
        return result

    # If initial spiral doesn't achieve optimum, try local search / simulated annealing
    # to find a configuration with more edges

    # If initial spiral doesn't achieve optimum, try local search / simulated annealing
    # to find a configuration with more edges
    
    # Precompute neighbor offsets for distance-1 checks
    # Neighbors at distance 1: (da,db) where da^2 + db^2 + da*db = 1
    # Check all small (da,db):
    # (1,0): 1+0+0=1 ✓
    # (0,1): 0+1+0=1 ✓
    # (-1,1): 1+1-1=1 ✓
    # (-1,0): 1+0+0=1 ✓
    # (0,-1): 0+1+0=1 ✓
    # (1,-1): 1+1-1=1 ✓
    # Others? (2,0): 4, no. (1,1): 1+1+1=3, no.
    # So the 6 neighbors are: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Build lookup for fast membership and edge counting
    def build_lookup(points):
        lookup = {}
        for p in points:
            lookup[(p[0], p[1])] = True
        return lookup
    
    def count_edges_fast(points):
        lookup = build_lookup(points)
        count = 0
        for p in points:
            a, b = p
            for da, db in neighbors:
                if (a+da, b+db) in lookup:
                    count += 1
        return count // 2  # Each edge counted twice
    
    # Local search: try moving each point to a nearby unoccupied position
    # to maximize edges
    
    def try_move(points, idx, new_a, new_b):
        # Check if new position is valid (not occupied, within bounds, distance >= 1 from all)
        a, b = points[idx]
        if new_a == a and new_b == b:
            return False
        if abs(new_a) > 64 or abs(new_b) > 64:
            return False
        
        old_pos = (a, b)
        new_pos = (new_a, new_b)
        
        # Check not occupied
        for p in points:
            if p[0] == new_a and p[1] == new_b:
                return False
        
        # Check distance >= 1 from all other points
        for j, p in enumerate(points):
            if j == idx:
                continue
            if dist_sq([new_a, new_b], p) < 1:
                return False
        return True
    
    # Simulated annealing / hill climbing
    best_points = [p[:] for p in result]
    best_score = count_edges_fast(best_points)
    
    current_points = [p[:] for p in result]
    current_score = count_edges_fast(current_points)
    
    # Temperature schedule
    temp = 1.0
    cooling = 0.999
    
    max_iters = 100000
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        if best_score >= target_edges:
            break
        
        # Pick a random point
        idx = random.randint(0, len(current_points) - 1)
        a, b = current_points[idx]
        
        # Generate a random nearby move
        # Choose a random direction and step size 1 or 2
        da = random.randint(-3, 3)
        db = random.randint(-3, 3)
        new_a = a + da
        new_b = b + db
        
        if not try_move(current_points, idx, new_a, new_b):
            continue
        
        # Compute score change
        old_score = current_score
        # Temporarily move
        current_points[idx] = [new_a, new_b]
        new_score = count_edges_fast(current_points)
        delta = new_score - old_score
        
        if delta >= 0 or random.random() < math.exp(delta / temp):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[idx] = [a, b]
        
        temp *= cooling
        if temp < 0.01:
            break
    
    if best_score > count_edges(result):
        result = best_points
    
    # Final verification
    final_edges = count_edges_fast(result)
    
    # Additional local search if we haven't hit the target
    if final_edges < target_edges:
        # Try a more focused local search: for each point, try all nearby empty positions
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for idx in range(len(result)):
                if time.time() > deadline - 5.0:
                    break
                a, b = result[idx]
                for da in range(-4, 5):
                    for db in range(-4, 5):
                        if time.time() > deadline - 5.0:
                            break
                        new_a = a + da
                        new_b = b + db
                        if new_a == a and new_b == b:
                            continue
                        if not try_move(result, idx, new_a, new_b):
                            continue
                        old_score = count_edges_fast(result)
                        result[idx] = [new_a, new_b]
                        new_score = count_edges_fast(result)
                        if new_score > old_score:
                            improved = True
                        else:
                            result[idx] = [a, b]
    
    return result
