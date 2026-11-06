import time
import random

def _count_ordinary_lines(points):
    # points: list of (x, y)
    # Returns number of lines containing exactly 2 points
    line_counts = {}
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            # Normalize line (a, b, c) for ax + by + c = 0
            # a = y1 - y2
            # b = x2 - x1
            # c = x1*y2 - x2*y1
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # Normalize to handle signs and gcd
            if a == 0 and b == 0:
                continue # Should not happen for distinct points
            
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            elif a == 0 and b == 0:
                pass
            else:
                # Make first non-zero positive
                if a < 0:
                    a, b, c = -a, -b, -c
                elif a == 0 and b < 0:
                    a, b, c = -a, -b, -c
            
            # Divide by gcd
            from math import gcd
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for count in line_counts.values():
        if count == 1: # Wait, line_counts counts pairs.
             # If a line has k points, it has C(k,2) pairs.
             # So if count == 1, it means C(k,2) == 1 => k=2.
             # So this line has exactly 2 points.
             ordinary += 1
             
    return ordinary

def construct():
    # Initial 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    # Convert to tuples for hashing and calculation
    pts = [tuple(p) for p in points]
    
    best_count = _count_ordinary_lines(pts)
    best_pts = pts[:]
    
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    # Local search
    # We will try to move points to nearby integer coordinates
    # to reduce the number of ordinary lines
    
    # To speed up, we can pre-compute or just iterate
    # Since n=12, pairs=66. Counting is fast.
    
    # Define a set of valid coordinates to avoid huge numbers
    # but we can just use small integers
    
    # Simple hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline: break
            x, y = best_pts[i]
            # Try neighbors
            for dx in [-5, -2, -1, 1, 2, 5]:
                for dy in [-5, -2, -1, 1, 2, 5]:
                    new_x = x + dx
                    new_y = y + dy
                    new_pt = (new_x, new_y)
                    
                    # Check distinctness
                    if new_pt in best_pts:
                        continue
                    if new_pt == best_pts[i]:
                        continue
                    
                    # Create new list
                    trial_pts = best_pts[:]
                    trial_pts[i] = new_pt
                    
                    # Check not all collinear
                    # Simple check: if all points lie on a line, return 0? No, invalid.
                    # We can check if all points are collinear by checking if all pairs form the same line.
                    # Or just count lines. If there is only 1 line, it's collinear.
                    # But _count_ordinary_lines returns 0 if all points are on one line?
                    # If all 12 points are on one line, C(12,2)=66 pairs, 1 line with 66 pairs.
                    # line_counts will have one key with value 66.
                    # ordinary = 0.
                    # But the problem says "not all collinear".
                    # So we must check this.
                    
                    # Quick collinearity check:
                    # Take first 3 points. If they are collinear, check if all others are on that line.
                    p0, p1, p2 = trial_pts[0], trial_pts[1], trial_pts[2]
                    # Line through p0, p1
                    a = p0[1] - p1[1]
                    b = p1[0] - p0[0]
                    c = p0[0]*p1[1] - p1[0]*p0[1]
                    collinear = True
                    for p in trial_pts:
                        if a*p[0] + b*p[1] + c != 0:
                            collinear = False
                            break
                    if collinear:
                        continue
                    
                    count = _count_ordinary_lines(trial_pts)
                    if count < best_count:
                        best_count = count
                        best_pts = trial_pts
                        improved = True
                        break # Restart outer loop or just continue?
                        # Usually, restarting or continuing is fine.
                        # Let's break to re-evaluate from new state.
                if improved:
                    break
            if improved:
                break
        
        # If no improvement, try a random kick
        if not improved:
             # Randomly move one point
             i = random.randint(0, 11)
             x, y = best_pts[i]
             new_x = x + random.randint(-10, 10)
             new_y = y + random.randint(-10, 10)
             new_pt = (new_x, new_y)
             if new_pt not in best_pts:
                 trial_pts = best_pts[:]
                 trial_pts[i] = new_pt
                 # Check collinear
                 p0, p1, p2 = trial_pts[0], trial_pts[1], trial_pts[2]
                 a = p0[1] - p1[1]
                 b = p1[0] - p0[0]
                 c = p0[0]*p1[1] - p1[0]*p0[1]
                 collinear = True
                 for p in trial_pts:
                     if a*p[0] + b*p[1] + c != 0:
                         collinear = False
                         break
                 if not collinear:
                     count = _count_ordinary_lines(trial_pts)
                     if count < best_count:
                         best_count = count
                         best_pts = trial_pts
                         improved = True
                     else:
                         # Accept with some probability? Or just reject.
                         # Simple hill climbing rejects.
                         pass

    return [list(p) for p in best_pts]
