import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial configuration: 4x4 grid
    # This gives 10 lines of 4 points.
    pts = [[x, y] for x in range(4) for y in range(4)]
    
    def evaluate(points):
        # Check validity
        # 1. Distinct
        if len(set(tuple(p) for p in points)) != 16:
            return -1000000
        # 2. Bounds
        for x, y in points:
            if abs(x) > 1000 or abs(y) > 1000:
                return -1000000
        
        # Count lines with exactly 4 points
        # And check for 5 collinear
        line_counts = defaultdict(int)
        
        # Use normalized line representation
        # For pair (x1,y1), (x2,y2):
        # A = y2 - y1
        # B = x1 - x2
        # C = x2*y1 - x1*y2
        # Normalize (A,B,C) by GCD
        
        for i in range(16):
            for j in range(i+1, 16):
                x1, y1 = points[i]
                x2, y2 = points[j]
                A = y2 - y1
                B = x1 - x2
                C = x2*y1 - x1*y2
                
                # Normalize
                if A == 0 and B == 0:
                    continue # Should not happen if distinct
                
                # Find GCD
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return abs(a)
                
                g = gcd(gcd(A, B), C)
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    # If A,B are 0, C must be 0 for distinct points? No.
                    # If A=0, B!=0, C can be anything.
                    # If A=0, B=0, then points are same.
                    pass
                
                # Ensure unique sign
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                elif A == 0 and B == 0:
                    # Degenerate, shouldn't happen
                    continue
                elif A == 0:
                    if B < 0:
                        A = -A; B = -B; C = -C
                    # If B > 0, ok
                else:
                    if A < 0:
                        A = -A; B = -B; C = -C
                    elif A == 0 and B < 0:
                        A = -A; B = -B; C = -C

                line_counts[(A, B, C)] += 1
        
        # The line_counts dictionary counts pairs.
        # A line with k points has k*(k-1)/2 pairs.
        # We want lines with exactly 4 points.
        # k=4 -> 6 pairs.
        # k=5 -> 10 pairs.
        
        n4_lines = 0
        n5_lines = 0
        for count in line_counts.values():
            # count is number of pairs on that line
            # k*(k-1)/2 = count
            # k=2 -> 1
            # k=3 -> 3
            # k=4 -> 6
            # k=5 -> 10
            if count == 6:
                n4_lines += 1
            elif count >= 10:
                # Check if it's exactly 5 or more
                # If count == 10, k=5.
                # If count > 10, k>5.
                n5_lines += 1
        
        if n5_lines > 0:
            return -1000000 + n4_lines # Penalize heavily
            
        return n4_lines

    best_score = evaluate(pts)
    best_pts = [p[:] for p in pts]
    
    current_score = best_score
    current_pts = [p[:] for p in pts]
    
    temperature = 10.0
    min_temperature = 0.01
    
    while time.time() < deadline:
        # Generate neighbor
        new_pts = [p[:] for p in current_pts]
        idx = random.randint(0, 15)
        if random.random() < 0.5:
            new_pts[idx][0] += random.randint(-3, 3)
        else:
            new_pts[idx][1] += random.randint(-3, 3)
            
        # Optional: random jump to escape local minima
        if random.random() < 0.05:
            idx = random.randint(0, 15)
            new_pts[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
            
        new_score = evaluate(new_pts)
        
        # Metropolis acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / temperature):
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = [p[:] for p in current_pts]
                
        # Cool down
        temperature *= 0.999
        if temperature < min_temperature:
            temperature = min_temperature
            
        # Restart if stuck?
        # Simple restart logic: if no improvement for X steps, restart from best
        # But with SA, cooling handles it.
        
    return best_pts
