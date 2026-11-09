import random
import time

def is_collinear(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

def has_4_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            count = 2
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return True
    return False

def max_no_3_collinear(points):
    n = len(points)
    # Precompute collinear masks
    # mask[i][j] is a bitmask of points k such that i, j, k are collinear
    mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            m = (1 << i) | (1 << j)
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    m |= (1 << k)
            mask[i][j] = m
            mask[j][i] = m
            
    best = 0
    
    def backtrack(idx, current_mask, size):
        nonlocal best
        if size + (n - idx) <= best:
            return
        if idx == n:
            best = max(best, size)
            return
            
        # Option 1: Skip idx
        backtrack(idx + 1, current_mask, size)
        
        # Option 2: Add idx
        # Check if idx can be added
        # For all a in current_mask, mask[idx][a] & current_mask must be 0?
        # No, if a is in current_mask, and b is in current_mask, and a,b,idx collinear, then invalid.
        # So we need: for all a in current_mask, (mask[idx][a] & current_mask) == 0?
        # If mask[idx][a] contains b, and b is in current_mask, then idx,a,b are collinear.
        # So yes.
        
        can_add = True
        cm = current_mask
        while cm:
            lsb = cm & -cm
            a = lsb.bit_length() - 1
            if (mask[idx][a] & current_mask) != 0:
                can_add = False
                break
            cm ^= lsb
            
        if can_add:
            backtrack(idx + 1, current_mask | (1 << idx), size + 1)

    backtrack(0, 0, 0)
    return best

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    random.seed(42)
    
    def random_points():
        pts = set()
        while len(pts) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.add((x, y))
        return list(pts)
    
    best_points = random_points()
    while has_4_collinear(best_points):
        best_points = random_points()
        
    best_score = max_no_3_collinear(best_points)
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, 19)
        old_x, old_y = best_points[i]
        new_x = random.randint(-50, 50)
        new_y = random.randint(-50, 50)
        
        if (new_x, new_y) in best_points:
            continue
            
        new_points = best_points.copy()
        new_points[i] = (new_x, new_y)
        
        if has_4_collinear(new_points):
            continue
            
        score = max_no_3_collinear(new_points)
        
        if score < best_score:
            best_score = score
            best_points = new_points
            if best_score <= 8:
                break
                
    return [[x, y] for x, y in best_points]
