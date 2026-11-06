import time
import random
import math
from itertools import combinations

def _count_four_point_lines(points):
    """Count lines containing exactly 4 of the points."""
    n = len(points)
    count = 0
    seen_lines = set()
    
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            g = math.gcd(dx, dy)
            if g != 0:
                dxn = dx // g
                dyn = dy // g
            else:
                dxn, dyn = 0, 0
            
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
                
            a = dyn
            b = -dxn
            c = a * p1[0] + b * p1[1]
            
            line_key = (a, b, c)
            
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            pts_on_line = []
            for k in range(n):
                pk = points[k]
                if a * pk[0] + b * pk[1] == c:
                    pts_on_line.append(k)
                    
            if len(pts_on_line) == 4:
                count += 1
            elif len(pts_on_line) >= 5:
                return -1
    return count

def _is_valid(points):
    """Check that no five points are collinear."""
    n = len(points)
    seen = set()
    for p in points:
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
        
    seen_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(dx, dy)
            if g != 0:
                dxn = dx // g
                dyn = dy // g
            else:
                continue
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            a = dyn
            b = -dxn
            c = a * p1[0] + b * p1[1]
            line_key = (a, b, c)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            
            cnt = 0
            for k in range(n):
                pk = points[k]
                if a * pk[0] + b * pk[1] == c:
                    cnt += 1
            if cnt >= 5:
                return False
    return True

def _evaluate(points):
    """Returns score, or -1 if invalid."""
    if not _is_valid(points):
        return -1
    return _count_four_point_lines(points)

def _local_search(points, grid_points, time_limit, rng):
    """Perform local search by trying all single-point moves."""
    n = len(points)
    current_score = _evaluate(points)
    if current_score < 0:
        return points, current_score
    
    # Build a set of occupied points for fast lookup
    occupied = set((p[0], p[1]) for p in points)
    
    # Try multiple passes of local search
    for pass_num in range(100):
        if time.time() > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_limit:
                break
            
            old_point = points[i][:]
            # Try each candidate point
            for candidate in grid_points:
                cx, cy = candidate[0], candidate[1]
                if (cx, cy) in occupied:
                    continue
                
                points[i] = [cx, cy]
                new_score = _evaluate(points)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    break
                elif new_score < 0:
                    # Invalid, revert
                    points[i] = old_point
                else:
                    # Same score, keep trying other candidates
                    points[i] = old_point
            
            if improved:
                break
        
        if not improved:
            break
    
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Define a larger grid of candidate points
    # Use a 10x10 grid centered at 0 for more diversity
    grid_points = []
    for x in range(-5, 6):
        for y in range(-5, 6):
            grid_points.append([x, y])
    
    # Also add some spread-out points
    for x in range(-50, 51, 10):
        for y in range(-50, 51, 10):
            if [x, y] not in grid_points:
                grid_points.append([x, y])
    
    random.seed(42)
    best_points = None
    best_score = -1
    
    # Known good starting configurations to try
    known_configs = [
        # 4x4 grid
        [[x, y] for x in range(4) for y in range(4)],
        # 4x4 grid shifted
        [[x+1, y+1] for x in range(4) for y in range(4)],
        # 2x8 configuration
        [[x, y] for x in range(8) for y in range(2)],
        # 8x2 configuration  
        [[x, y] for x in range(2) for y in range(8)],
    ]
    
    # Try known configs first
    for config in known_configs:
        if time.time() > deadline - 10:
            break
        score = _evaluate(config)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # Random restarts with simulated annealing
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        # Select 16 random points from the grid
        current_points = [p[:] for p in random.sample(grid_points, 16)]
        
        # Initial evaluation
        current_score = _evaluate(current_points)
        if current_score < 0:
            continue
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
        
        # Simulated Annealing
        T = 3.0
        T_min = 0.005
        cooling = 0.998
        
        while T > T_min and time.time() < deadline - 5:
            # Pick a point to move
            i = random.randint(0, 15)
            old_point = current_points[i][:]
            
            # Pick a new point from the grid
            new_point = None
            for _ in range(200):
                candidate = random.choice(grid_points)
                if candidate not in current_points:
                    new_point = candidate
                    break
            
            if new_point is None:
                continue
                
            current_points[i] = new_point[:]
            
            new_score = _evaluate(current_points)
            
            if new_score < 0:
                delta = -50
                accept = False
            else:
                delta = new_score - current_score
                if delta > 0:
                    accept = True
                else:
                    if random.random() < math.exp(delta / T):
                        accept = True
                    else:
                        accept = False
            
            if accept:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[i] = old_point
            
            T *= cooling
        
        # Local search refinement
        if time.time() < deadline - 5:
            refined_points, refined_score = _local_search(
                [p[:] for p in current_points], grid_points, deadline - 5, random
            )
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points
    
    # Final local search on best points
    if best_points is not None and time.time() < deadline - 2:
        refined_points, refined_score = _local_search(
            [p[:] for p in best_points], grid_points, deadline - 2, random
        )
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    
    if best_points is None:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
