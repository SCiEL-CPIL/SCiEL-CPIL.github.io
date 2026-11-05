import time
import random
from math import gcd

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    ordinary = 0
    for key, pair_count in lines.items():
        if pair_count == 1:
            ordinary += 1
            
    return ordinary

def greedy_construct(base_points, deadline, search_range=15):
    """
    Greedily add points to the base configuration to minimize ordinary lines.
    """
    current_points = [p[:] for p in base_points]
    current_score = count_ordinary_lines(current_points)
    
    # Candidate positions for new points
    # We search in a box around the existing points
    min_x = min(p[0] for p in current_points) - search_range
    max_x = max(p[0] for p in current_points) + search_range
    min_y = min(p[1] for p in current_points) - search_range
    max_y = max(p[1] for p in current_points) + search_range
    
    # Limit search space for efficiency
    candidates = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if abs(x) <= 10**9 and abs(y) <= 10**9:
                candidates.append([x, y])
                
    # Shuffle candidates to avoid bias, but keep some structure
    random.shuffle(candidates)
    
    for _ in range(12 - len(current_points)):
        best_pt = None
        best_score = current_score
        # Time check
        if time.time() > deadline:
            break
            
        for pt in candidates:
            if any(pt == p for p in current_points):
                continue
            new_points = current_points + [pt]
            new_score = count_ordinary_lines(new_points)
            if new_score < best_score:
                best_score = new_score
                best_pt = pt
        if best_pt is not None:
            current_points.append(best_pt)
            current_score = best_score
        else:
            # If no improvement found in the box, expand or break
            # For simplicity, break if no better point found
            break
            
    return current_points, current_score

def local_search(points, deadline, max_iter=2000):
    """
    Local search to minimize ordinary lines.
    """
    n = len(points)
    best_points = [p[:] for p in points]
    best_score = count_ordinary_lines(best_points)
    
    # Precompute a set of candidate moves
    small_steps = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if (dx, dy) != (0, 0)]
    large_steps = [(dx, dy) for dx in range(-20, 21, 2) for dy in range(-20, 21, 2) if (dx, dy) != (0, 0)]
    
    iters = 0
    while iters < max_iter and time.time() < deadline:
        iters += 1
        improved = False
        
        for i in range(n):
            x, y = best_points[i]
            
            # Try small steps
            for dx, dy in small_steps:
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                if any(new_pt == p for p in best_points):
                    continue
                new_points = best_points[:]
                new_points[i] = new_pt
                new_score = count_ordinary_lines(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
                    improved = True
                    break
            if improved:
                break
            
            # Try large steps
            for dx, dy in large_steps:
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                new_pt = [nx, ny]
                if any(new_pt == p for p in best_points):
                    continue
                new_points = best_points[:]
                new_points[i] = new_pt
                new_score = count_ordinary_lines(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Random jump
            if random.random() < 0.7:
                i = random.randint(0, n-1)
                x, y = best_points[i]
                nx = x + random.randint(-50, 50)
                ny = y + random.randint(-50, 50)
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    new_pt = [nx, ny]
                    if not any(new_pt == p for p in best_points):
                        new_points = best_points[:]
                        new_points[i] = new_pt
                        new_score = count_ordinary_lines(new_points)
                        if new_score <= best_score:
                            best_score = new_score
                            best_points = new_points
            else:
                break
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_points = None
    best_score = float('inf')
    
    # Base configurations
    base33 = [[x, y] for x in range(3) for y in range(3)]
    base43 = [[x, y] for x in range(4) for y in range(3)]
    base34 = [[x, y] for x in range(3) for y in range(4)]
    
    # 1. Greedy construction from 3x3
    pts, score = greedy_construct(base33, deadline, search_range=10)
    if score < best_score:
        best_score = score
        best_points = pts
        
    # 2. Greedy construction from 4x3
    pts, score = greedy_construct(base43, deadline, search_range=10)
    if score < best_score:
        best_score = score
        best_points = pts
        
    # 3. Greedy construction from 3x4
    pts, score = greedy_construct(base34, deadline, deadline, search_range=10)
    if score < best_score:
        best_score = score
        best_points = pts
        
    # 4. Local search from the best greedy result
    if best_points:
        pts, score = local_search(best_points, deadline, max_iter=1000)
        if score < best_score:
            best_score = score
            best_points = pts
            
    # 5. Local search from original configs (fallback)
    configs = [
        base33 + [[3, 0], [0, 3], [3, 3]],
        base33 + [[-1, 1], [1, -1], [4, 2]]
    ]
    for cfg in configs:
        if time.time() > deadline - 5:
            break
        pts, score = local_search(cfg, deadline, max_iter=500)
        if score < best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        best_points = base33 + [[3, 0], [0, 3], [3, 3]]
        
    return best_points
