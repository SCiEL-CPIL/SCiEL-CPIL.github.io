import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + da*db + db*db

    def count_edges(points):
        n = len(points)
        edges = 0
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    # Theoretical optimum for n=32 is 76.
    # Harborth's theorem: floor(3n - sqrt(12n - 3)) = floor(96 - sqrt(381)) = floor(96 - 19.519) = 76
    
    # Strategy: Directly construct hexagonal clusters and verify they achieve 76 edges.
    # A hexagonal cluster of side lengths (a, b, c, a, b, c) has 3(a+b+c) - 3 points? No.
    # Number of points in a hexagon with side lengths a, b, c is 3(a+b+c) - 3?
    # Actually, the number of points in a hexagon with side lengths a, b, c, a, b, c is:
    # 1 + 3(a+b+c) - 3? No.
    # Let's use the formula: N = 3(a+b+c) - 3?
    # For a=1, b=1, c=1 (hexagon of 7 points): 3(3)-3 = 6? No, it's 7.
    # Formula: N = 3(a+b+c) - 3 is for perimeter?
    # Let's just generate hexagons and check.
    
    # A hexagon with side lengths a, b, c, a, b, c has:
    # N = 3(a+b+c) - 3? No.
    # N = 1 + 3(a+b+c) - 3?
    # Let's check: a=1, b=1, c=1 -> 1+3(3)-3 = 7. Correct.
    # a=2, b=2, c=2 -> 1+3(6)-3 = 16. Correct (hexagon of side 2 has 16 points).
    # a=1, b=1, c=2 -> 1+3(4)-3 = 10.
    # We need N=32.
    # 3(a+b+c) - 2 = 32 => 3(a+b+c) = 34, not integer.
    # So a perfect hexagon cannot have exactly 32 points.
    # We need to add or remove points from a hexagon.
    
    # Let's try hexagon with a=1, b=2, c=3: N = 1+3(6)-3 = 16. Too small.
    # a=2, b=2, c=3: N = 1+3(7)-3 = 19.
    # a=2, b=3, c=3: N = 1+3(8)-3 = 22.
    # a=3, b=3, c=3: N = 1+3(9)-3 = 25.
    # a=3, b=3, c=4: N = 1+3(10)-3 = 28.
    # a=3, b=4, c=4: N = 1+3(11)-3 = 31. Close!
    # a=4, b=4, c=4: N = 1+3(12)-3 = 34. Too big.
    
    # So a hexagon with sides (3,4,4) has 31 points. We can add 1 point to make 32.
    # Or a hexagon with sides (2,4,5) has N = 1+3(11)-3 = 31.
    # Or (1,5,5) -> 1+3(11)-3 = 31.
    
    # Let's generate the hexagon with sides (3,4,4) and try adding a point.
    # Or generate (4,4,4) and remove 2 points.
    
    # Let's write a function to generate a hexagon.
    def gen_hex(a, b, c):
        points = set()
        # Center at (0,0)
        # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # We can build it by rows.
        # Row y from -c to a+b-1?
        # Let's use the standard construction.
        # For y in range(-c, a+b):
        #   x_min = max(-a, y-c, -b-y)
        #   x_max = min(b, a-y, c-y)
        #   for x in range(x_min, x_max+1):
        #       points.add((x,y))
        
        for y in range(-c, a+b):
            x_min = max(-a, y-c, -b-y)
            x_max = min(b, a-y, c-y)
            for x in range(x_min, x_max+1):
                points.add((x,y))
        
        return list(points)

    # Try (3,4,4) -> 31 points
    hex31 = gen_hex(3, 4, 4)
    if len(hex31) != 31:
        # Fallback
        hex31 = gen_hex(4, 4, 4)
        # Remove 2 points
        hex31 = hex31[:32]
        
    # Now try to add 1 point to hex31 to get 32 points with max edges.
    # Or just use local search on a 32-point set.
    
    # Let's try a different approach: Start with a 32-point set from a hexagon and refine.
    # Hexagon (4,4,4) has 34 points. Remove 2 points to get 32.
    hex34 = gen_hex(4, 4, 4)
    
    # Try all pairs of points to remove from hex34 and see which gives max edges.
    best_points = None
    best_edges = 0
    
    # This might be too slow if hex34 is large. 34 choose 2 = 561.
    # count_edges is O(32^2) = 1024 ops. 561 * 1024 ~ 574k ops. Very fast.
    
    for i in range(len(hex34)):
        for j in range(i+1, len(hex34)):
            # Skip removing i and j
            pts = [p for k, p in enumerate(hex34) if k != i and k != j]
            if len(pts) != 32:
                continue
            e = count_edges(pts)
            if e > best_edges:
                best_edges = e
                best_points = [list(p) for p in pts]
            if best_edges == 76:
                break
        if best_edges == 76:
            break
    
    # If we didn't find 76, try adding 1 point to hex31
    if best_edges < 76:
        # Try adding a point to hex31
        # Find a point adjacent to many points in hex31 that is not in hex31
        hex31_set = set(tuple(p) for p in hex31)
        candidates = []
        for p in hex31:
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                np = (p[0]+dx, p[1]+dy)
                if np not in hex31_set:
                    candidates.append(np)
        
        # Try each candidate
        for cand in candidates:
            pts = [list(p) for p in hex31] + [list(cand)]
            if len(pts) != 32:
                continue
            e = count_edges(pts)
            if e > best_edges:
                best_edges = e
                best_points = [list(p) for p in pts]
            if best_edges == 76:
                break
    
    # If still not 76, do local search
    if best_points is None:
        # Fallback to a simple grid
        best_points = [[i, j] for i in range(4) for j in range(8)]
        best_edges = count_edges(best_points)
    
    if best_edges < 76:
        import random
        random.seed(42)
        current = [list(p) for p in best_points]
        current_edges = best_edges
        
        # Local search: move one point to a random empty neighbor
        for iter in range(2000):
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            for i in range(32):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ox, oy = current[i][0], current[i][1]
                # Try moving to all 6 neighbors
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 64 or abs(ny) > 64:
                        continue
                    # Check if (nx, ny) is occupied
                    occupied = False
                    for k, p in enumerate(current):
                        if k != i and p[0] == nx and p[1] == ny:
                            occupied = True
                            break
                    if occupied:
                        continue
                    
                    # Check distance to all others
                    valid = True
                    for k in range(32):
                        if k == i: continue
                        if dist_sq([nx, ny], current[k]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Try the move
                    current[i][0] = nx
                    current[i][1] = ny
                    new_edges = count_edges(current)
                    if new_edges > current_edges:
                        current_edges = new_edges
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_points = [list(p) for p in current]
                    else:
                        # Revert
                        current[i][0] = ox
                        current[i][1] = oy
                
                if best_edges == 76:
                    break
            if best_edges == 76:
                break
            if not improved:
                # Try random restarts or perturbations if stuck
                if random.random() < 0.05:
                    idx = random.randint(0, 31)
                    ox, oy = current[idx][0], current[idx][1]
                    # Move to a random nearby spot
                    for _ in range(10):
                        dx = random.randint(-2, 2)
                        dy = random.randint(-2, 2)
                        nx, ny = ox + dx, oy + dy
                        if abs(nx) > 64 or abs(ny) > 64: continue
                        occupied = any(k != idx and p[0]==nx and p[1]==ny for k, p in enumerate(current))
                        if occupied: continue
                        valid = all(dist_sq([nx, ny], current[k]) >= 1 for k in range(32) if k != idx)
                        if not valid: continue
                        current[idx][0] = nx
                        current[idx][1] = ny
                        new_edges = count_edges(current)
                        if new_edges > current_edges:
                            current_edges = new_edges
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_points = [list(p) for p in current]
                        else:
                            current[idx][0] = ox
                            current[idx][1] = oy
                        break

    # Fallback if something went wrong
    if best_points is None:
        return [[a, b] for a in range(16) for b in range(2)]
        
    return best_points
