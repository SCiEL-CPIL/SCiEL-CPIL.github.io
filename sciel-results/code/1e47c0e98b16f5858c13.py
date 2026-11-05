import time
import random
import itertools

# EVOLVE-BLOCK-START
def _count_lines(points):
    """
    Count the number of lines containing exactly 4 points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 4:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            A = dy
            B = -dx
            C = -(A * x1 + B * y1)
            
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g == 0:
                continue
            
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in line_counts:
                line_counts[key] = {i, j}
            else:
                line_counts[key].update({i, j})
                
    count = 0
    for key, pts_in_line in line_counts.items():
        if len(pts_in_line) == 4:
            count += 1
            
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    best_score = _count_lines(points)
    best_points = [p[:] for p in points]
    
    def local_search(current_points, max_iters=20):
        pts = [p[:] for p in current_points]
        occupied = {tuple(p) for p in pts}
        score = _count_lines(pts)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            best_move_score = score
            best_move_idx = -1
            best_move_pos = None
            
            for i in range(16):
                orig_x, orig_y = pts[i]
                orig_tuple = (orig_x, orig_y)
                
                # Remove current point from occupied set
                occupied.remove(orig_tuple)
                
                # Generate candidates
                candidates = set()
                # Small neighborhood
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx == 0 and dy == 0: continue
                        nx, ny = orig_x + dx, orig_y + dy
                        if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                            if (nx, ny) not in occupied:
                                candidates.add((nx, ny))
                
                # Random jumps
                for _ in range(20):
                    nx = orig_x + random.randint(-50, 50)
                    ny = orig_y + random.randint(-50, 50)
                    if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                        if (nx, ny) not in occupied:
                            candidates.add((nx, ny))
                
                for nx, ny in candidates:
                    pts[i] = [nx, ny]
                    s = _count_lines(pts)
                    if s > best_move_score:
                        best_move_score = s
                        best_move_idx = i
                        best_move_pos = (nx, ny)
                    
                    pts[i] = [orig_x, orig_y]
                
                # Restore occupied
                occupied.add(orig_tuple)
            
            if best_move_idx != -1:
                old_x, old_y = pts[best_move_idx]
                occupied.remove((old_x, old_y))
                pts[best_move_idx] = [best_move_pos[0], best_move_pos[1]]
                occupied.add(best_move_pos)
                score = best_move_score
                improved = True
                
        return score, pts

    while time.time() < deadline:
        # Perform local search from current points
        score, pts = local_search(points, max_iters=20)
        
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            points = [p[:] for p in pts]
        
        # Check time
        if time.time() > deadline - 5.0:
            break
        
        # Restart: Perturb the configuration significantly
        # Move 4 random points to random locations
        for _ in range(4):
            idx = random.randint(0, 15)
            # Generate a random point far away
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
            # Ensure distinctness
            valid = True
            for k in range(16):
                if k != idx and points[k][0] == new_x and points[k][1] == new_y:
                    valid = False
                    break
            
            if valid:
                points[idx] = [new_x, new_y]
            else:
                # If collision, try a few times or just skip
                pass
                
    return best_points
# EVOLVE-BLOCK-END
