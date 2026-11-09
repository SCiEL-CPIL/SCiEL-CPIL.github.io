"""Rewritten program for the suite task; uses Simulated Annealing on small integer grids."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 points.
    Points are a list of [x, y].
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    # Precompute pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line coefficients A, B, C for Ax + By + C = 0
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            # Normalize
            if A == 0 and B == 0:
                continue
            
            g = gcd(abs(A), abs(B))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            # Canonical sign
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            elif A == 0 and B == 0 and C < 0:
                C = -C
            elif A == 0 and B == 0 and C == 0:
                # Should not happen for distinct points, but just in case
                A = 1
                
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def _is_valid(points):
    """Check validity: distinct, integer, bounds, not all collinear."""
    if len(points) != 12:
        return False
    
    seen = set()
    for p in points:
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
        
    # Check not all collinear
    # Pick first two distinct points
    p0 = points[0]
    p1 = points[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    
    for i in range(2, 12):
        pi = points[i]
        if dx * (pi[1] - p0[1]) != dy * (pi[0] - p0[0]):
            return True
            
    return False

def _local_search(points, max_time=10.0):
    """
    Perform local search (Steepest Descent with random restarts within the call) 
    to minimize ordinary lines.
    """
    start_time = time.time()
    current_points = [list(p) for p in points]
    current_score = _count_ordinary_lines(current_points)
    best_points = [list(p) for p in current_points]
    best_score = current_score
    
    # Small window for moves
    window = 5
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try moving each point
        for i in range(12):
            x, y = current_points[i]
            
            # Try all positions in a small box around current
            for dx in range(-window, window + 1):
                for dy in range(-window, window + 1):
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx = x + dx
                    ny = y + dy
                    
                    # Bounds check (loose check, final check later)
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    
                    # Check collision
                    collision = False
                    for j in range(12):
                        if j == i:
                            continue
                        if current_points[j][0] == nx and current_points[j][1] == ny:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    # Apply move
                    old = current_points[i]
                    current_points[i] = [nx, ny]
                    
                    # Quick validity check (not all collinear is hard to check fast, 
                    # but if we move one point, it's likely not all collinear unless we 
                    # are in a very specific bad state. We can skip full validity check 
                    # for speed and rely on the fact that we start valid and moves are 
                    # small. However, if we move a point to make all collinear, it's invalid.
                    # But "not all collinear" is a constraint. 
                    # Actually, if we minimize ordinary lines, we want many collinearities.
                    # If all 12 become collinear, ordinary lines = 0? No, the line has 12 points, 
                    # so 0 ordinary lines. But the task says "not all collinear".
                    # So we must ensure not all collinear.
                    
                    # Efficient check: if current state was not all collinear, 
                    # moving one point might make it all collinear only if the other 11 
                    # are collinear and the new point lies on that line.
                    # This is a rare event in search, but possible.
                    # We will check validity only if score improves significantly or periodically.
                    # For now, assume valid unless we detect all collinear.
                    
                    new_score = _count_ordinary_lines(current_points)
                    
                    # Check if all collinear
                    if new_score == 0:
                        # If score is 0, it could be 0 ordinary lines because all are on one line (invalid)
                        # or a perfect configuration.
                        # Check if all 12 are on one line.
                        is_all_collinear = True
                        p0 = current_points[0]
                        p1 = current_points[1]
                        dx = p1[0] - p0[0]
                        dy = p1[1] - p0[1]
                        for k in range(2, 12):
                            pk = current_points[k]
                            if dx * (pk[1] - p0[1]) != dy * (pk[0] - p0[0]):
                                is_all_collinear = False
                                break
                        if is_all_collinear:
                            current_points[i] = old
                            continue

                    if new_score < current_score:
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [list(p) for p in current_points]
                        improved = True
                        # Break out of dx/dy loop for this i to re-evaluate from new position
                        break
                    else:
                        current_points[i] = old
                
                if improved:
                    break
        
        if not improved:
            # No improvement found in this pass. 
            # Try a random kick to escape local minimum
            i = random.randint(0, 11)
            x, y = current_points[i]
            for _ in range(10):
                dx = random.randint(-window, window)
                dy = random.randint(-window, window)
                nx = x + dx
                ny = y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                collision = False
                for j in range(12):
                    if j == i:
                        continue
                    if current_points[j][0] == nx and current_points[j][1] == ny:
                        collision = True
                        break
                if collision:
                    continue
                
                old = current_points[i]
                current_points[i] = [nx, ny]
                
                # Check validity
                if not _is_valid(current_points):
                    current_points[i] = old
                    continue
                    
                new_score = _count_ordinary_lines(current_points)
                # Accept worse moves with some probability (Simulated Annealing style)
                # Simple random walk for now
                current_points[i] = [nx, ny]
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current_points]
                break
            # If kick failed or didn't happen, break out of while loop
            # We'll just break to restart with a new initial config in the outer loop
            break
            
    return best_points, best_score

def construct():
    """
    Construct 12 integer points with minimized ordinary lines.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Initial configurations to try
    # 1. 4x3 Grid
    grid_4x3 = [[i, j] for i in range(4) for j in range(3)]
    
    # 2. 3x4 Grid
    grid_3x4 = [[i, j] for i in range(3) for j in range(4)]
    
    # 3. Hexagonal-ish cluster
    hex_cluster = [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [1, 3], [2, 3], [3, 3]
    ]
    
    # 4. Random small integers
    random.seed(42)
    random_config = []
    used = set()
    while len(random_config) < 12:
        x = random.randint(-5, 5)
        y = random.randint(-5, 5)
        if (x, y) not in used:
            used.add((x, y))
            random_config.append([x, y])
            
    initial_configs = [grid_4x3, grid_3x4, hex_cluster, random_config]
    
    # Add some more random restarts
    for _ in range(5):
        rc = []
        used = set()
        while len(rc) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in used:
                used.add((x, y))
                rc.append([x, y])
        initial_configs.append(rc)
        
    for config in initial_configs:
        if time.time() > deadline:
            break
            
        # Run local search for a fraction of the time
        # Divide remaining time among configs
        remaining = deadline - time.time()
        time_per_config = min(10.0, remaining / len(initial_configs))
        
        pts, score = _local_search(config, max_time=time_per_config)
        
        if score < best_score:
            best_score = score
            best_points = pts
            
        if best_score == 6:
            # Theoretical minimum, stop early
            break
            
    # If no valid points found (should not happen), return a default valid set
    if best_points is None:
        best_points = grid_4x3
        
    # Final validation
    if not _is_valid(best_points):
        # Fallback
        best_points = grid_4x3
        
    return best_points
# EVOLVE-BLOCK-END
