"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def count_ordinary_lines_fast(points):
    """
    Count lines containing exactly 2 of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None
        
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        g = math.gcd(abs(dx), abs(dy))
        if g > 0:
            dx //= g
            dy //= g
            
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def point_on_line(p, line_key):
        if line_key is None:
            return False
        dx, dy, c = line_key
        x, y = p
        return dy * x - dx * y == c

    unique_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is not None:
                unique_lines.add(key)
                
    ordinary_count = 0
    for line_key in unique_lines:
        count = 0
        for p in points:
            if point_on_line(p, line_key):
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_count = float('inf')
    
    def update_best(points, count):
        nonlocal best_points, best_count
        if count < best_count:
            best_count = count
            best_points = [p[:] for p in points]
            
    # 1. Baseline: 3x4 grid
    grid_points = [[x, y] for x in range(4) for y in range(3)]
    grid_count = count_ordinary_lines_fast(grid_points)
    update_best(grid_points, grid_count)
    
    # 2. Simulated Annealing with Line-Snapping Moves
    # The key insight from the inspiration is that "line-snapping" moves are effective.
    # A move consists of taking a point P_i and moving it to a new position P'_i such that 
    # P'_i lies on a line defined by two other points P_j, P_k.
    # This directly creates or reinforces a 3-point line, reducing ordinary lines.
    
    def get_line_snapping_moves(points, i):
        """
        Generate candidate moves for point i that place it on a line defined by two other points.
        Returns a list of (new_x, new_y) candidates.
        """
        moves = []
        n = len(points)
        # To limit computation, we only consider lines defined by pairs (j, k) where j, k != i
        # and we look for integer intersections or points on that line within a reasonable range.
        
        # We can't iterate over all pairs (j,k) for every i in every step if it's too slow, 
        # but n=12 is small. 11*10/2 = 55 pairs.
        for j in range(n):
            if j == i: continue
            for k in range(j+1, n):
                if k == i: continue
                p_j = points[j]
                p_k = points[k]
                
                x1, y1 = p_j
                x2, y2 = p_k
                dx = x2 - x1
                dy = y2 - y1
                
                if dx == 0 and dy == 0: continue
                
                # Normalize direction
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                g = math.gcd(abs(dx), abs(dy))
                if g == 0: continue
                dx //= g
                dy //= g
                
                # Line equation: dy*x - dx*y = c
                c = dy * x1 - dx * y1
                
                # We want to find integer points (x, y) on this line.
                # We will sample a few positions along the line.
                # Since we want to keep coordinates smallish (to encourage more collisions), 
                # we'll look near the existing points or just sample a bounded region.
                
                # Let's try to place the new point such that it's close to the current point i
                # or close to the segment jk.
                # Actually, a better strategy for SA is to propose a random point on the line 
                # within a box around the current configuration.
                
                # Let's just pick a few random integer points on this line.
                # If dx != 0, x can be any integer, y = (dy*x - c) / dx.
                # If dy != 0, y can be any integer, x = (dy*y - c) / dy.
                
                # To ensure we find valid integer points quickly, we can just try to shift
                # point i to lie on this line by adjusting one coordinate.
                
                # If dx != 0:
                #   y = (dy * x - c) / dx
                #   We need dy * x - c to be divisible by dx.
                #   Let x = x_i + delta.
                #   This is a bit complex to solve generally for a specific delta.
                
                # Alternative: Just propose the intersection of the line with the horizontal/vertical 
                # through the current point? No, that might not be integer.
                
                # Simpler: Propose points on the line that are "near" the current point i.
                # Let's just sample x values around x_i and check if y is integer.
                x_i, y_i = points[i]
                
                candidates = []
                # Try x in range [x_i - 10, x_i + 10]
                if dx != 0:
                    for x_test in range(x_i - 10, x_i + 11):
                        val = dy * x_test - c
                        if val % dx == 0:
                            y_test = val // dx
                            if abs(x_test) <= 10**9 and abs(y_test) <= 10**9:
                                # Ensure it's not the same as j or k or i
                                if (x_test, y_test) != (x1, y1) and (x_test, y_test) != (x2, y2) and (x_test, y_test) != (x_i, y_i):
                                    candidates.append([x_test, y_test])
                elif dy != 0:
                    # dx == 0, vertical line x = x1
                    # y can be anything.
                    for y_test in range(y_i - 10, y_i + 11):
                        if (x1, y_test) != (x1, y1) and (x1, y_test) != (x2, y2) and (x1, y_test) != (x_i, y_i):
                            candidates.append([x1, y_test])
                            
                moves.extend(candidates)
                
        return moves

    def simulated_annealing(initial_points, max_iter=5000):
        points = [p[:] for p in initial_points]
        current_cost = count_ordinary_lines_fast(points)
        
        # SA Parameters
        T_start = 1.0
        T_end = 0.01
        alpha = 0.999 # Cooling rate
        
        best_local_points = [p[:] for p in points]
        best_local_cost = current_cost
        
        for t in range(max_iter):
            if time.time() > deadline - 2.0:
                break
                
            T = T_start * (alpha ** t)
            if T < T_end:
                break
                
            # Choose a random point to move
            i = random.randint(0, 11)
            
            # Generate moves
            moves = get_line_snapping_moves(points, i)
            
            if not moves:
                # Fallback: random small move
                old_x, old_y = points[i]
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
                if abs(new_x) <= 10**9 and abs(new_y) <= 10**9:
                    moves = [[new_x, new_y]]
                else:
                    continue
                    
            # Pick a random move
            new_pos = random.choice(moves)
            
            # Check validity (distinct points)
            old_pos = points[i][:]
            points[i] = new_pos
            if len(set(tuple(p) for p in points)) < 12:
                points[i] = old_pos
                continue
                
            new_cost = count_ordinary_lines_fast(points)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                if current_cost < best_local_cost:
                    best_local_cost = current_cost
                    best_local_points = [p[:] for p in points]
            else:
                points[i] = old_pos
                
        return best_local_points, best_local_cost

    # Run SA from the grid
    sa_points, sa_cost = simulated_annealing(grid_points)
    update_best(sa_points, sa_cost)
    
    # Run SA from a few random starts
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
            
        # Random start in a small box
        pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
        if len(set(tuple(p) for p in pts)) < 12:
            continue
            
        sa_pts, sa_c = simulated_annealing(pts, max_iter=2000)
        update_best(sa_pts, sa_c)
        
        if best_count == 6:
            break

    # Final polish: Hill climbing on the best found
    if best_points:
        def hill_climb(points):
            pts = [p[:] for p in points]
            current_cost = count_ordinary_lines_fast(pts)
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                for i in range(12):
                    # Try all line-snapping moves for point i
                    moves = get_line_snapping_moves(pts, i)
                    # Also try small random moves
                    old_x, old_y = pts[i]
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx==0 and dy==0: continue
                            nx, ny = old_x + dx, old_y + dy
                            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                                moves.append([nx, ny])
                    
                    for move in moves:
                        old_pos = pts[i][:]
                        pts[i] = move
                        if len(set(tuple(p) for p in pts)) < 12:
                            pts[i] = old_pos
                            continue
                        new_cost = count_ordinary_lines_fast(pts)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                        else:
                            pts[i] = old_pos
            return pts, current_cost
            
        final_pts, final_cost = hill_climb(best_points)
        update_best(final_pts, final_cost)

    if best_points is None:
        return grid_points
        
    return best_points

# EVOLVE-BLOCK-END
