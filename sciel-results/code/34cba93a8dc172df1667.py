import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    N = 32
    # Directions for hexagonal lattice neighbors
    # (a, b) coordinates
    dirs = [
        (1, 0), (0, 1), (-1, 1), 
        (-1, 0), (0, -1), (1, -1)
    ]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    def count_edges(points):
        # points is a list of (a, b)
        # Returns number of pairs with dist_sq == 1
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def is_valid(points, idx, new_pos):
        # Check if moving points[idx] to new_pos is valid (no collisions)
        # We only need to check against other points, not itself
        a, b = new_pos
        # Check bounds if necessary, though lattice is infinite
        # But problem says |a|,|b| <= 64
        if abs(a) > 64 or abs(b) > 64:
            return False
        
        for j in range(len(points)):
            if j == idx:
                continue
            if dist_sq(new_pos, points[j]) < 1:
                return False
        return True

    def generate_spiral(n):
        # Generate a dense hexagonal spiral of n points
        # Start with (0,0)
        points = [(0, 0)]
        if n <= 1:
            return points[:n]
        
        # Spiral expansion
        # Directions: E, NE, NW, W, SW, SE
        # Step counts: 1, 1, 2, 2, 3, 3, 4, 4...
        # Or more simply, iterate over hexagon rings
        # Ring 0: (0,0)
        # Ring 1: 6 points
        # Ring 2: 12 points
        # Ring k: 6k points
        
        # Let's use a simple spiral walk
        # (0,0)
        # (1,0)
        # (1,-1) -> wait, let's stick to the standard spiral that fills hexagons
        # A better way: enumerate all lattice points by hex distance from origin
        # and sort them.
        
        candidates = []
        max_hex_dist = 10 # 1+6+12+18+24 = 61 > 32, so 4 rings is enough. 1+6+12+18=37.
        for hex_d in range(max_hex_dist + 1):
            if hex_d == 0:
                candidates.append((0,0))
                continue
            
            # Generate points on the hexagon of distance hex_d
            # Start at (hex_d, 0)
            a, b = hex_d, 0
            for side in range(6):
                for step in range(hex_d):
                    candidates.append((a, b))
                    # Move to next point on this side
                    da, db = dirs[side]
                    a += da
                    b += db
            
        # Sort by hex distance, then angle to make it compact
        def hex_dist(p):
            a, b = p
            return max(abs(a), abs(b), abs(-a-b))
            
        # Actually, the spiral generation above might have duplicates or order issues.
        # Let's just collect all points within a certain range and pick the densest cluster?
        # No, let's use the explicit ring generation which is cleaner.
        
        # Ring 0: (0,0)
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2: starts at (2,0)...
        
        points = []
        # Add ring 0
        points.append((0,0))
        
        current_ring = 1
        while len(points) < n:
            # Add points on current_ring
            # Start at (current_ring, 0)
            a, b = current_ring, 0
            for side in range(6):
                for step in range(current_ring):
                    if len(points) < n:
                        points.append((a, b))
                    da, db = dirs[side]
                    a += da
                    b += db
            current_ring += 1
            
        return points[:n]

    # Initial solution
    best_points = generate_spiral(N)
    best_score = count_edges(best_points)
    
    # Local Search
    # We will try to improve the score by moving individual points.
    # Since the spiral is already very good, we might just need to tweak a few.
    # Hill climbing with random restarts if stuck.
    
    import copy
    
    current_points = copy.deepcopy(best_points)
    current_score = best_score
    
    # Pre-compute neighbors for speed? 
    # dist_sq is O(1). Count is O(N^2) = 32^2 = 1024 ops. Very fast.
    
    # We can do a simple hill climbing:
    # 1. Pick random index
    # 2. Pick random small move (delta a, delta b)
    # 3. If valid, calculate new score.
    # 4. If new score > current score, accept.
    # 5. If no improvement for X steps, try a random restart or larger moves.
    
    # To ensure we find 76, we can also just check if we already have 76.
    # If not, search.
    
    max_iterations = 100000
    stuck_counter = 0
    max_stuck = 5000
    
    while time.time() < deadline and stuck_counter < max_stuck:
        if current_score >= 76:
            break
        
        idx = random.randint(0, N - 1)
        # Random move within a small radius
        # Radius 2 or 3 is usually enough for local fixes
        ra = random.randint(-3, 3)
        rb = random.randint(-3, 3)
        
        new_a = current_points[idx][0] + ra
        new_b = current_points[idx][1] + rb
        new_pos = (new_a, new_b)
        
        if is_valid(current_points, idx, new_pos):
            # Calculate new score
            # Optimization: only recalculate edges involving idx
            # Old edges involving idx:
            old_edges_with_idx = 0
            for j in range(N):
                if j != idx and dist_sq(current_points[idx], current_points[j]) == 1:
                    old_edges_with_idx += 1
            
            # New edges involving idx:
            new_edges_with_idx = 0
            for j in range(N):
                if j != idx and dist_sq(new_pos, current_points[j]) == 1:
                    new_edges_with_idx += 1
            
            delta = new_edges_with_idx - old_edges_with_idx
            if delta > 0:
                current_points[idx] = new_pos
                current_score += delta
                stuck_counter = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = copy.deepcopy(current_points)
            else:
                stuck_counter += 1
        else:
            stuck_counter += 1
            
        # Optional: Simulated Annealing-like random move if stuck for a while
        if stuck_counter > max_stuck // 2 and random.random() < 0.01:
             # Allow a move that doesn't improve score if it's "close"
             # Or just reset to best and try different random seed
             pass

    # If we didn't reach 76, try a more aggressive search or just return best found
    # Given the spiral is usually optimal, this should rarely trigger
    if best_score < 76:
         # Try a few random restarts with different initial configurations?
         # Or just accept best. The spiral is very likely 76.
         pass

    # Ensure we return exactly 32 distinct integer pairs
    # Convert to list of lists
    result = [[a, b] for a, b in best_points]
    
    # Final verification
    if len(result) != 32:
        # Fallback if something went wrong
        result = [[a, b] for a, b in generate_spiral(32)]
        
    return result

# To ensure reproducibility in testing but randomness in search, we can seed based on time
# But for the evaluator, it just runs once.
# The random module is seeded by system time by default.

if __name__ == "__main__":
    # Simple test
    res = construct()
    # print(len(res))
    # edges = 0
    # for i in range(32):
    #     for j in range(i+1, 32):
    #         a1, b1 = res[i]
    #         a2, b2 = res[j]
    #         da = a1 - a2
    #         db = b1 - b2
    #         d2 = da*da + da*db + db*db
    #         if d2 == 1:
    #             edges += 1
    #         elif d2 < 1:
    #             print("Collision!")
    # print("Edges:", edges)
    pass
