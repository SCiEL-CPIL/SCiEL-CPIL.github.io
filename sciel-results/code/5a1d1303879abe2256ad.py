import time
import random
from math import gcd

def normalize_line(p1, p2):
    """Return canonical line key from two points."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 1, x1)
    if dy == 0:
        return (1, 0, y1)
    g = gcd(abs(dx), abs(dy))
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0:
        dx_n = -dx_n
        dy_n = -dy_n
    c = dy_n * x1 - dx_n * y1
    return (dx_n, dy_n, c)

def count_4_lines_fast(points):
    """Fast count of lines with exactly 4 points. Returns -1 if any 5 collinear."""
    n = len(points)
    if n != 16:
        return 0
    
    # Use a dict mapping line key -> bitmask of points on that line
    line_mask = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = normalize_line(points[i], points[j])
            if key is None:
                continue
            mask = (1 << i) | (1 << j)
            if key in line_mask:
                line_mask[key] |= mask
            else:
                line_mask[key] = mask
    
    count4 = 0
    for key, mask in line_mask.items():
        popcount = bin(mask).count('1')
        if popcount >= 5:
            return -1
        if popcount == 4:
            count4 += 1
    return count4

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with the 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in grid]
    best_score = count_4_lines_fast(best_points)
    
    random.seed(42)
    
    # Local search function
    def local_search(current, current_score, max_steps=2000):
        for step in range(max_steps):
            if time.time() > deadline:
                return current, current_score
            
            # Pick a random point to move
            idx = random.randint(0, 15)
            old = current[idx][:]
            
            # Try a random new position in a local neighborhood
            # We expand the neighborhood size occasionally
            radius = random.choice([2, 3, 5, 10, 20])
            nx = old[0] + random.randint(-radius, radius)
            ny = old[1] + random.randint(-radius, radius)
            
            if abs(nx) > 1000 or abs(ny) > 1000:
                continue
                
            current[idx] = [nx, ny]
            
            # Check distinctness
            pts_set = set(tuple(p) for p in current)
            if len(pts_set) < 16:
                current[idx] = old
                continue
                
            new_score = count_4_lines_fast(current)
            if new_score < 0:
                current[idx] = old
                continue
                
            if new_score > current_score:
                current_score = new_score
                # Accept the move
            else:
                # Revert the move
                current[idx] = old
                
            # If we've done many steps without improvement, break early to allow restart
            # This is a simple heuristic. We track improvements externally.
        return current, current_score

    # Iterated Local Search
    current = [p[:] for p in best_points]
    current_score = best_score
    
    no_improve_counter = 0
    
    while time.time() < deadline:
        # Perform local search
        current, current_score = local_search(current, current_score, max_steps=500)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current]
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            
        # If stuck, perturb the current solution
        if no_improve_counter > 5:
            # Perturb: move 2-3 random points significantly
            for _ in range(random.randint(2, 3)):
                idx = random.randint(0, 15)
                current[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
            
            # Ensure distinctness after perturbation
            for _ in range(100):
                if len(set(tuple(p) for p in current)) == 16:
                    break
                idx = random.randint(0, 15)
                current[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
            
            current_score = count_4_lines_fast(current)
            if current_score < 0:
                # Invalid, revert to best
                current = [p[:] for p in best_points]
                current_score = best_score
            no_improve_counter = 0

    return best_points
