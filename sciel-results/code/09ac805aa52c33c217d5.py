import random
import time
import math
from collections import defaultdict

def _no_four_collinear(points):
    n = len(points)
    if n < 4:
        return True
        
    def line_rep(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0) else 1
        if g == 0:
            g = 1
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
        
    line_counts = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            rep = line_rep(i, j)
            line_counts[rep] += 1
            
    for count in line_counts.values():
        if count > 3:
            return False
    return True

def _largest_no_three_collinear(points):
    n = len(points)
    collinear_mask = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask
            collinear_mask[j][i] = mask
            
    best = [0]
    
    def backtrack(selected_indices, next_idx, current_size):
        if current_size + (n - next_idx) <= best[0]:
            return
        if next_idx >= n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        backtrack(selected_indices, next_idx + 1, current_size)
        
        p = next_idx
        valid = True
        for a in range(len(selected_indices)):
            for b in range(a + 1, len(selected_indices)):
                i = selected_indices[a]
                j = selected_indices[b]
                if collinear_mask[i][j] & (1 << p):
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            backtrack(selected_indices + [p], next_idx + 1, current_size + 1)
            
    backtrack([], 0, 0)
    return best[0]

def construct():
    random.seed(42)
    
    # Start with random points
    initial = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
    # Ensure distinct
    while len(set(map(tuple, initial))) < 20:
        idx = random.randint(0, 19)
        initial[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        
    # Ensure no 4 collinear
    # This might take a while if random, but 20 points in 200x200 is sparse.
    # Actually, probability of 4 collinear is very low.
    # But to be safe, we can check and fix.
    if not _no_four_collinear(initial):
        # Try to fix by moving points
        for _ in range(1000):
            if _no_four_collinear(initial):
                break
            idx = random.randint(0, 19)
            initial[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            if len(set(map(tuple, initial))) < 20:
                # Revert or retry
                initial[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        if not _no_four_collinear(initial):
            # Fallback to parabola if random fails (unlikely)
            initial = [[x, x * x] for x in range(-10, 10)]
            
    current_points = [p[:] for p in initial]
    current_score = _largest_no_three_collinear(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    start_time = time.time()
    max_time = 110
    
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() - start_time < max_time:
        # Decide move type
        # 50% random, 50% collinearity-seeking
        if random.random() < 0.5:
            # Random move
            i = random.randint(0, 19)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            new_points = [p[:] for p in current_points]
            new_points[i] = [new_x, new_y]
        else:
            # Collinearity-seeking move
            # Pick 2 anchor points
            i, j = random.sample(range(20), 2)
            x1, y1 = current_points[i]
            x2, y2 = current_points[j]
            
            # Find a point k to move
            k = random.choice([x for x in range(20) if x != i and x != j])
            
            # Find a position for k that is collinear with i, j
            # Line through (x1, y1) and (x2, y2)
            # Parametric: x = x1 + t*(x2-x1), y = y1 + t*(y2-y1)
            # We need integer t such that x, y are integers and in range.
            # Also, the point (x, y) must not be i or j.
            # And it must not create a 4-collinear.
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Find gcd to step by primitive vector
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g == 0:
                g = 1
            step_x = dx // g
            step_y = dy // g
            
            # Try several t values
            # t=1 is j, t=0 is i.
            # Try t = -1, 2, -2, 3, etc.
            candidate_pos = None
            for t in [-1, 2, -2, 3, -3, 4, -4, 5, -5]:
                nx = x1 + t * step_x
                ny = y1 + t * step_y
                if -100 <= nx <= 100 and -100 <= ny <= 100:
                    # Check if this position is occupied by another point
                    # If it is, we can't move k there unless we swap? 
                    # For now, just skip if occupied.
                    occupied = False
                    for m in range(20):
                        if m == k: continue
                        if current_points[m][0] == nx and current_points[m][1] == ny:
                            occupied = True
                            break
                    if not occupied:
                        candidate_pos = [nx, ny]
                        break
                        
            if candidate_pos:
                new_points = [p[:] for p in current_points]
                new_points[k] = candidate_pos
            else:
                # Fallback to random move
                i = random.randint(0, 19)
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                new_points = [p[:] for p in current_points]
                new_points[i] = [new_x, new_y]
                
        # Check validity
        if len(set(map(tuple, new_points))) != 20:
            continue
        if not _no_four_collinear(new_points):
            continue
            
        new_score = _largest_no_three_collinear(new_points)
        
        # Acceptance
        if new_score < current_score:
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            delta = new_score - current_score
            if delta < 0:
                current_points = new_points
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_points = new_points
                    current_score = new_score
        
        T *= cooling_rate
        
    return best_points
