"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    # We want 32 points on triangular lattice maximizing unit-distance pairs
    # Distance between (a,b) and (a',b') is sqrt((a-a')^2 + (a-a')(b-b') + (b-b')^2)
    # Unit distance means: (a-a')^2 + (a-a')(b-b') + (b-b')^2 == 1
    
    def count_unit_edges(points):
        point_set = set(points)
        count = 0
        # 6 neighbors in axial coordinates (a, b)
        # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
        for a, b in points:
            if (a+1, b) in point_set: count += 1
            if (a, b+1) in point_set: count += 1
            if (a-1, b+1) in point_set: count += 1
            if (a-1, b) in point_set: count += 1
            if (a, b-1) in point_set: count += 1
            if (a+1, b-1) in point_set: count += 1
        return count // 2
    
    # Start with a hexagonal spiral arrangement
    # Generate hexagonal spiral points
    def hex_spiral(n_points):
        points = [(0, 0)]
        # Directions in triangular lattice (6 neighbors)
        # In (a,b) coords with omega = exp(i*pi/3):
        # Neighbors are at: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Actually let me use the standard hex spiral
        # Layer 0: (0,0)
        # Layer 1: 6 points
        # Layer k: 6k points
        
        # Standard hexagonal spiral in axial coordinates
        # Using cube-like generation
        points = []
        if n_points <= 1:
            return [(0,0)]
        
        # Generate in spiral order
        # Center
        points.append((0, 0))
        if len(points) >= n_points:
            return points[:n_points]
        
        # For each ring r >= 1, add 6r points
        r = 1
        while len(points) < n_points:
            # Start at (r, 0) and go around
            # In axial coords, ring r consists of 6r points
            # Starting point: (r, 0)
            # The 6 sides go in directions that cycle through the 6 neighbors
            
            # Generate all points in ring r
            ring_points = []
            # Side 1: from (r, 0) to (0, r) - direction (-1, 1)
            for i in range(r):
                a = r - i
                b = i
                ring_points.append((a, b))
            # Side 2: from (0, r) to (-r, 0) - direction (-1, 0)
            for i in range(1, r):
                a = -i
                b = r
                ring_points.append((a, b))
            # Side 3: from (-r, 0) to (-r, -r) - direction (0, -1)
            for i in range(1, r):
                a = -r
                b = -i
                ring_points.append((a, b))
            # Side 4: from (-r, -r) to (0, -r) - direction (1, -1)
            for i in range(1, r):
                a = -r + i
                b = -r
                ring_points.append((a, b))
            # Side 5: from (0, -r) to (r, -r) - direction (1, 0)
            for i in range(1, r):
                a = i
                b = -r
                ring_points.append((a, b))
            # Side 6: from (r, -r) to (r, 0) - direction (0, 1)
            for i in range(1, r):
                a = r
                b = -r + i
                ring_points.append((a, b))
            
            for p in ring_points:
                if len(points) >= n_points:
                    break
                points.append(p)
            
            r += 1
        
        return points[:n_points]
    
    # Get hex spiral for 32 points
    points = hex_spiral(32)
    best_score = count_unit_edges(points)
    best_points = points[:]
    
    if best_score == 76:
        return [[p[0], p[1]] for p in best_points]

    import random
    point_set = set(points)
    
    # Local search: Random Restart / Simulated Annealing style
    # We will try to move points to nearby empty spots
    # To escape local optima, we allow moves that don't decrease score (or slightly decrease with low prob)
    # But since we have a hard optimum 76, we can just do First Improvement with random restarts if stuck
    
    current_score = best_score
    current_points = points[:]
    current_set = point_set
    
    # Pre-calculate neighbor offsets
    offsets = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Define a helper to calculate score change efficiently?
    # Since N=32, full recalculation is fast (~192 checks).
    # Let's just use the fast count_unit_edges
    
    stall_counter = 0
    
    while time.time() < deadline - 5:
        # Pick a random point to move
        i = random.randint(0, 31)
        a_i, b_i = current_points[i]
        
        # Try to move it to a random nearby empty spot
        # Search radius 2 or 3
        found_move = False
        for _ in range(20): # Try a few random moves
            a_new = a_i + random.randint(-2, 2)
            b_new = b_i + random.randint(-2, 2)
            
            if (a_new, b_new) == (a_i, b_i):
                continue
            if (a_new, b_new) in current_set:
                continue
            if abs(a_new) > 64 or abs(b_new) > 64:
                continue
            
            # Perform move
            current_set.remove((a_i, b_i))
            current_set.add((a_new, b_new))
            current_points[i] = (a_new, b_new)
            
            new_score = count_unit_edges(current_points)
            
            if new_score >= current_score:
                # Accept improvement or equal (to allow wandering)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points[:]
                        if best_score == 76:
                            return [[p[0], p[1]] for p in best_points]
                    stall_counter = 0
                else:
                    # Equal score, accept with some probability to escape plateaus
                    if random.random() < 0.1:
                        stall_counter = 0
                    else:
                        # Revert
                        current_set.remove((a_new, b_new))
                        current_set.add((a_i, b_i))
                        current_points[i] = (a_i, b_i)
                        stall_counter += 1
                found_move = True
                break
            else:
                # Worse score, revert (unless we want SA, but let's stick to greedy+restart for simplicity first)
                current_set.remove((a_new, b_new))
                current_set.add((a_i, b_i))
                current_points[i] = (a_i, b_i)
        
        if not found_move:
            stall_counter += 1
        
        # If stuck for too long, restart from best_points
        if stall_counter > 100:
            current_points = best_points[:]
            current_set = set(current_points)
            current_score = best_score
            stall_counter = 0
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
