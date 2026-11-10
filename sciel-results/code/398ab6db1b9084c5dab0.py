import time
import random
from collections import Counter

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area among all triples."""
    n = len(points)
    areas = []
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    return max(Counter(areas).values())

def _score_with_moved(points, idx, new_point):
    """Compute max multiplicity after moving point idx to new_point, efficiently."""
    n = len(points)
    new_points = list(points)
    new_points[idx] = new_point
    
    # Areas not involving idx: recompute is expensive, so let's just do full recompute
    # But for 30 points it's 4060 triples, which is fast enough
    areas = []
    for i in range(n):
        x1, y1 = new_points[i]
        for j in range(i+1, n):
            x2, y2 = new_points[j]
            for k in range(j+1, n):
                x3, y3 = new_points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    return max(Counter(areas).values())

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate candidate positions
    candidates = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Initial configurations
    configs = []
    
    # 6x5 grid centered
    for dx in range(-2, 4):
        for dy in range(-2, 3):
            configs.append([(dx, dy) for dx in range(-2, 4) for dy in range(-2, 3)])
    # Wait, that's 6x5=30 but let me be more careful
    configs = []
    
    # Various grid shapes
    for (rows, cols) in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15), (30,1), (1,30), (5,4), (4,5)]:
        if rows * cols != 30:
            continue
        # Center the grid
        x0 = -(cols - 1) // 2
        y0 = -(rows - 1) // 2
        pts = [(x0 + x, y0 + y) for x in range(cols) for y in range(rows)]
        # Check bounds
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            configs.append(pts)
    
    # Also try some non-rectangular grids
    # Parallelogram-like
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                for d in range(-3, 4):
                    if abs(a*d - b*c) == 0:
                        continue
                    pts = []
                    for i in range(6):
                        for j in range(5):
                            x = i * a + j * c
                            y = i * b + j * d
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                pts.append((x, y))
                    if len(pts) == 30 and len(set(pts)) == 30:
                        configs.append(pts)
    
    if not configs:
        # Fallback
        return [(x, y) for x in range(6) for y in range(5)]
    
    # Evaluate all initial configs
    best_score = 0
    best_config = configs[0]
    for cfg in configs:
        if time.time() > deadline:
            break
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_config = list(cfg)
    
    # Local search
    random.seed(42)
    current = list(best_config)
    current_score = best_score
    
    # Precompute set of current points for O(1) lookup
    current_set = set(current)
    
    # Local search: try moving each point to nearby positions
    max_iters = 5000
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and iter_count < max_iters and no_improve_count < max_no_improve:
        iter_count += 1
        improved = False
        
        # Randomly pick a point to move
        idx = random.randint(0, 29)
        old_point = current[idx]
        
        # Generate candidate moves: nearby points
        ox, oy = old_point
        moves = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                np = (ox + dx, oy + dy)
                if -30 <= np[0] <= 30 and -30 <= np[1] <= 30:
                    if np not in current_set:
                        moves.append(np)
        
        if not moves:
            no_improve_count += 1
            continue
        
        random.shuffle(moves)
        best_move = None
        best_move_score = current_score
        
        for np in moves[:20]:  # Limit moves tried per iteration
            if time.time() > deadline:
                break
            # Temporarily move
            current[idx] = np
            current_set.discard(old_point)
            current_set.add(np)
            
            s = _score(current)
            
            current[idx] = old_point
            current_set.discard(np)
            current_set.add(old_point)
            
            if s > best_move_score:
                best_move_score = s
                best_move = np
        
        if best_move is not None:
            current[idx] = best_move
            current_set.discard(old_point)
            current_set.add(best_move)
            current_score = best_move_score
            best_score = max(best_score, current_score)
            best_config = list(current)
            no_improve_count = 0
            improved = True
        else:
            no_improve_count += 1
    
    # Also try swap-based local search
    swap_no_improve = 0
    max_swap_no_improve = 100
    while time.time() < deadline and swap_no_improve < max_swap_no_improve:
        i = random.randint(0, 29)
        j = random.randint(0, 29)
        if i == j:
            continue
        
        # Try moving point i to a random valid position
        ox, oy = current[i]
        found = False
        for _ in range(10):
            if time.time() > deadline:
                break
            np = (random.randint(-30, 30), random.randint(-30, 30))
            if np not in current_set:
                current[i] = np
                current_set.discard((ox, oy))
                current_set.add(np)
                s = _score(current)
                if s > current_score:
                    current_score = s
                    best_score = max(best_score, current_score)
                    best_config = list(current)
                    swap_no_improve = 0
                    found = True
                else:
                    current[i] = (ox, oy)
                    current_set.discard(np)
                    current_set.add((ox, oy))
        
        if not found:
            swap_no_improve += 1
    
    return best_config
