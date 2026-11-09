import random
import math
import time

_START_TIME = None
_DEADLINE = 145.0 # Safety margin

def _time_left():
    if _START_TIME is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _START_TIME)

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 2
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 4:
                        return True
    return False

def _max_3_free_subset(points):
    n = len(points)
    if n <= 2:
        return n
    
    collinearity = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and _is_collinear(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinearity[i][j] = mask
            collinearity[j][i] = mask
            
    best = [0]
    
    def backtrack(i, subset_mask, size):
        if size > best[0]:
            best[0] = size
        if i == n:
            return
        if size + (n - i) <= best[0]:
            return
        
        # Exclude
        backtrack(i + 1, subset_mask, size)
        
        # Include
        can_add = True
        temp_mask = subset_mask
        while temp_mask:
            a_bit = temp_mask & (-temp_mask)
            a = a_bit.bit_length() - 1
            if collinearity[a][i] & subset_mask:
                can_add = False
                break
            temp_mask ^= a_bit
            
        if can_add:
            backtrack(i + 1, subset_mask | (1 << i), size + 1)
            
    backtrack(0, 0, 0)
    return best[0]

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)
    
    best_score = 20
    best_points = None
    
    # Try multiple initializations
    for restart in range(10):
        if _time_left() < 20.0:
            break
            
        # Initialization: Perturbed Grid
        points = []
        base_x = [0, 10, 20, 30, 40]
        base_y = [0, 10, 20, 30]
        
        for x in base_x:
            for y in base_y:
                # Perturb
                px = x + random.randint(-3, 3)
                py = y + random.randint(-3, 3)
                # Keep within bounds
                px = max(-100, min(100, px))
                py = max(-100, min(100, py))
                points.append((px, py))
                
        # Ensure distinct
        # Simple repair
        valid = True
        for i in range(20):
            for j in range(i+1, 20):
                if points[i] == points[j]:
                    points[j] = (points[j][0] + 1, points[j][1])
        
        # Ensure no 4 collinear
        attempts = 0
        while _has_four_collinear(points) and attempts < 1000:
            idx = random.randint(0, 19)
            old = points[idx]
            points[idx] = (old[0] + random.randint(-5, 5), old[1] + random.randint(-5, 5))
            attempts += 1
            
        if _has_four_collinear(points):
            continue
            
        current_score = _max_3_free_subset(points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = list(points)
            
        # Local Search
        temperature = 1.0
        no_improve = 0
        max_no_improve = 50
        
        while _time_left() > 5.0 and no_improve < max_no_improve:
            idx = random.randint(0, 19)
            old_point = points[idx]
            
            # Generate new point
            # 50% chance: Random nearby
            # 50% chance: Targeted (on a line)
            if random.random() < 0.5:
                new_point = (old_point[0] + random.randint(-5, 5), old_point[1] + random.randint(-5, 5))
            else:
                # Pick two other points
                j, k = random.sample([i for i in range(20) if i != idx], 2)
                p1 = points[j]
                p2 = points[k]
                # Find a point on line p1-p2
                # dx, dy
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Try a few steps along the line
                found = False
                for step in range(-10, 11):
                    if step == 0: continue
                    # Parametric: P = P1 + t * (P2-P1)
                    # We want integer coordinates.
                    # If dx, dy have gcd g, then steps of g are integer.
                    # Simplified: Just try P1 + step * (dx/g, dy/g)
                    g = math.gcd(abs(dx), abs(dy)) if dx != 0 or dy != 0 else 1
                    if g == 0: continue
                    sx = dx // g
                    sy = dy // g
                    nx = p1[0] + step * sx
                    ny = p1[1] + step * sy
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        new_point = (nx, ny)
                        found = True
                        break
                if not found:
                    new_point = (old_point[0] + random.randint(-5, 5), old_point[1] + random.randint(-5, 5))
                    
            if new_point in points:
                continue
                
            points[idx] = new_point
            
            if _has_four_collinear(points):
                points[idx] = old_point
                continue
                
            new_score = _max_3_free_subset(points)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_points = list(points)
            else:
                # Accept with prob exp(-delta/T) ?
                # Since we want to minimize, and scores are small integers,
                # we mostly reject worse moves.
                if random.random() < math.exp((current_score - new_score) / temperature):
                    pass # Accept
                else:
                    points[idx] = old_point
                    no_improve += 1
                    
            temperature *= 0.95
            
    if best_points is None:
        # Fallback
        best_points = [(i, i*i) for i in range(20)]
        # Fix validity if needed (parabola has no 3 collinear? No, y=x^2 has no 3 collinear. So score is 20.)
        
    return [list(p) for p in best_points]
