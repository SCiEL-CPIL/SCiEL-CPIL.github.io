import random
import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 140.0  # 10s margin
    
    # Directions in axial coordinates (a, b)
    # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def get_neighbors(p):
        a, b = p
        return [(a+da, b+db) for da, db in dirs]
    
    def count_edges(points_set):
        count = 0
        for p in points_set:
            for n in get_neighbors(p):
                if n in points_set:
                    count += 1
        return count // 2
    
    # Initialize with a hexagon of radius 3 (37 points)
    # Axial coordinates for hexagon radius R:
    # -R <= a <= R
    # -R <= b <= R
    # -R <= a+b <= R  (Wait, this is for cube coords or specific projection)
    # In axial coords (a,b) for triangular lattice:
    # A hexagon of radius R centered at (0,0) includes points (a,b) such that
    # max(|a|, |b|, |a+b|) <= R ? No.
    # Distance in axial coords is not simply max.
    # Let's generate points by expanding rings.
    
    def generate_hexagon(R):
        points = set()
        points.add((0,0))
        for r in range(1, R+1):
            # Start at (r, 0) and go around
            # Directions to move along the perimeter
            # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
            # The 6 corners are:
            # (r, 0), (0, r), (-r, r), (-r, 0), (0, -r), (r, -r)
            # Let's just iterate through a bounding box and check distance.
            # Distance squared in axial coords: a^2 + ab + b^2
            # But "hexagon radius" usually means graph distance.
            # Graph distance in triangular lattice is not Euclidean.
            # However, for small R, we can just take all points with Euclidean distance <= R*sqrt(3) or similar?
            # Actually, the set of points at graph distance <= R from origin is a hexagon.
            # Let's just generate all points in a square of size 2R+1 and filter by graph distance?
            # Or simpler: BFS from (0,0) up to depth R.
            
            # Since R is small (3), we can just hardcode or use a simple expansion.
            # Let's use BFS.
            pass
            
    # Simpler initialization:
    # Start with a cluster of 32 points.
    # We'll use a "greedy" seed: place points one by one to maximize edges.
    # Or just start with a random set of 32 points in a small area and anneal.
    
    # Let's use a fixed seed for reproducibility in testing, but random in production?
    # The evaluator might run multiple times. We want to be robust.
    # Let's use multiple restarts.
    
    best_score = -1
    best_points = None
    
    for restart in range(50):
        # Initialize points
        # Start with a compact shape.
        # Generate a hexagon of radius 3 (37 points)
        hex_points = set()
        # Manually construct hexagon of radius 3
        # Center (0,0)
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2: ...
        # Let's just use a square of points and pick 32 closest to origin?
        # No, that might not be optimal shape.
        
        # Let's generate all points in range [-4, 4] and pick 32 randomly?
        # Better: Start with all points in a hexagon of radius 3.
        # Axial coords for hex radius 3:
        # (a,b) such that |a|<=3, |b|<=3, |a+b|<=3?
        # Let's check (3,0). a=3,b=0. a+b=3. OK.
        # (0,3). a=0,b=3. a+b=3. OK.
        # (3,3). a=3,b=3. a+b=6. NO.
        # So condition is max(|a|, |b|, |a+b|) <= R?
        # Let's test (2,1). a=2,b=1. a+b=3. Max=3. OK.
        # (2,-1). a=2,b=-1. a+b=1. Max=2. OK.
        # This defines a hexagon.
        
        hex3 = set()
        for a in range(-3, 4):
            for b in range(-3, 4):
                if max(abs(a), abs(b), abs(a+b)) <= 3:
                    hex3.add((a,b))
                    
        # hex3 should have 37 points.
        # Let's verify count.
        # R=0: 1
        # R=1: 7
        # R=2: 19
        # R=3: 37
        # Formula 3R(R+1)+1.
        # 3*3*4+1 = 37. Correct.
        
        if len(hex3) != 37:
             # Fallback if logic is wrong
             hex3 = set((a,b) for a in range(-4,5) for b in range(-4,5))
             # Sort by distance to origin and take 32?
             # Distance squared a^2+ab+b^2
             sorted_hex = sorted(hex3, key=lambda p: p[0]**2 + p[0]*p[1] + p[1]**2)
             points = set(sorted_hex[:32])
        else:
            # Remove 5 random points
            hex_list = list(hex3)
            # To get a good start, remove points with lowest degree in the induced subgraph?
            # Or just random.
            random.shuffle(hex_list)
            # Keep first 32
            points = set(hex_list[:32])
            
        current_points = points
        current_edges = count_edges(current_points)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        # Number of steps
        # We have ~10s per restart? No, 140s total.
        # 50 restarts is too many if each takes 2s.
        # Let's do 5 restarts with 20000 steps each.
        # Or 1 restart with 100000 steps.
        # Let's do 10 restarts, 5000 steps each.
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            if T < min_T:
                break
                
            p = random.choice(list(current_points))
            # Pick a random neighbor to move to
            # Or pick a random direction
            da, db = random.choice(dirs)
            new_p = (p[0]+da, p[1]+db)
            
            if new_p in current_points:
                continue
            
            # Check bounds (loose)
            if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
                continue
                
            # Calculate delta E
            # Remove p, add new_p
            # Old neighbors of p in current_points
            old_neighbors = [n for n in get_neighbors(p) if n in current_points]
            # New neighbors of new_p in (current_points - {p})
            # Note: p is in current_points, so we exclude it.
            new_neighbors = [n for n in get_neighbors(new_p) if n in current_points and n != p]
            
            delta_e = len(new_neighbors) - len(old_neighbors)
            
            if delta_e > 0 or random.random() < math.exp(delta_e / T):
                current_points.remove(p)
                current_points.add(new_p)
                current_edges += delta_e
                
                if current_edges > best_score:
                    best_score = current_edges
                    best_points = list(current_points)
                
                T *= cooling
        
        if time.time() - start_time > time_limit:
            break
            
    if best_points is None:
        # Fallback: return a simple valid config
        return [[a, b] for a in range(16) for b in range(2)]
        
    return best_points

# EVOLVE-BLOCK-END
