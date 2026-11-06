import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds, safety margin before 150s kill
    
    def score_points(pts):
        """Count lines with exactly 4 points, return (count, has_five)."""
        if len(pts) != 16:
            return 0, True
        # Check distinct
        seen = set()
        for p in pts:
            if (p[0], p[1]) in seen:
                return 0, True
            seen.add((p[0], p[1]))
        # For each pair, find the line, count points on it
        lines = {}  # line_key -> count
        for i in range(16):
            for j in range(i+1, 16):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                # Normalize direction: make first non-zero component positive
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Line key: direction + a point on the line (normalized)
                # Use the "cross product" form: a*x + b*y = c where (a,b) = (dy_n, -dx_n)
                a = dy_n
                b = -dx_n
                c = a * x1 + b * y1
                # Normalize (a, b, c)
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
        
        count4 = 0
        has_five = False
        for key, idx_set in lines.items():
            n = len(idx_set)
            if n == 4:
                count4 += 1
            elif n >= 5:
                has_five = True
        return count4, has_five
    
    def fast_score(pts):
        """Faster scoring using precomputed approach."""
        # Use a different approach: for each point pair, compute line and count
        # But we can cache: group by line
        line_points = {}
        for i in range(16):
            xi, yi = pts[i]
            for j in range(i+1, 16):
                xj, yj = pts[j]
                dx = xj - xi
                dy = yj - yi
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                a = dy_n
                b = -dx_n
                c = a * xi + b * yi
                g2 = gcd(gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                key = (a, b, c)
                if key not in line_points:
                    line_points[key] = set()
                line_points[key].add(i)
        
        count4 = 0
        has_five = False
        for key, idx_set in line_points.items():
            n = len(idx_set)
            if n == 4:
                count4 += 1
            elif n >= 5:
                has_five = True
        return count4, has_five
    
    # Start with 4x4 grid
    best_pts = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = fast_score(best_pts)
    
    # Try some initial configurations
    random.seed(42)
    
    # Hill climbing with random restarts
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            current_pts = [p[:] for p in best_pts]
        else:
            # Random restart: 4x4 grid with random perturbations
            current_pts = []
            for x in range(4):
                for y in range(4):
                    px = x + random.randint(-3, 3)
                    py = y + random.randint(-3, 3)
                    current_pts.append([px, py])
        
        cur_score, cur_has5 = fast_score(current_pts)
        cur_valid = not cur_has5
        if cur_valid and cur_score > best_score:
            best_score = cur_score
            best_pts = [p[:] for p in current_pts]
        
        # Hill climbing
        for step in range(3000):
            if time.time() - start_time > DEADLINE:
                break
            
            # Pick a random point and perturb it
            idx = random.randint(0, 15)
            old_x, old_y = current_pts[idx]
            # Random new position
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            # Check distinct
            current_pts[idx] = [new_x, new_y]
            valid = True
            for k in range(16):
                if k == idx:
                    continue
                if current_pts[k][0] == new_x and current_pts[k][1] == new_y:
                    valid = False
                    break
            if not valid:
                current_pts[idx] = [old_x, old_y]
                continue
            
            new_score, new_has5 = fast_score(current_pts)
            new_valid = not new_has5
            
            if new_valid and (new_score > cur_score or (new_score == cur_score and random.random() < 0.3)):
                cur_score = new_score
                cur_valid = True
                if cur_score > best_score:
                    best_score = cur_score
                    best_pts = [p[:] for p in current_pts]
            else:
                current_pts[idx] = [old_x, old_y]
        
        # Occasionally try a larger jump
        if restart % 5 == 4:
            for step in range(500):
                if time.time() - start_time > DEADLINE:
                    break
                idx = random.randint(0, 15)
                old_x, old_y = current_pts[idx]
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                current_pts[idx] = [new_x, new_y]
                valid = True
                for k in range(16):
                    if k == idx:
                        continue
                    if current_pts[k][0] == new_x and current_pts[k][1] == new_y:
                        valid = False
                        break
                if not valid:
                    current_pts[idx] = [old_x, old_y]
                    continue
                new_score, new_has5 = fast_score(current_pts)
                new_valid = not new_has5
                if new_valid and new_score > cur_score:
                    cur_score = new_score
                    cur_valid = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_pts = [p[:] for p in current_pts]
                else:
                    current_pts[idx] = [old_x, old_y]
    
    # Final validation
    final_score, final_has5 = fast_score(best_pts)
    if final_has5:
        # Fallback to 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts
