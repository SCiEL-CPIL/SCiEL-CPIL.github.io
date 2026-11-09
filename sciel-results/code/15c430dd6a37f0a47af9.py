import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin + buffer
    
    best_score = -1
    best_points = None
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0:
            if B < 0:
                A, B, C = -A, -B, -C
            elif B == 0:
                if C < 0:
                    A, B, C = -A, -B, -C
        return (A, B, C)

    def random_grid_init():
        offset_x = random.randint(-50, 50)
        offset_y = random.randint(-50, 50)
        scale = random.randint(1, 3)
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append((i * scale + offset_x, j * scale + offset_y))
        return pts

    def random_init():
        pts = []
        while len(pts) < 16:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in pts:
                pts.append((x, y))
        return pts

    def hill_climb(points):
        current_points = [tuple(p) for p in points]
        n = len(current_points)
        
        # Build initial line map
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(current_points[i], current_points[j])
                line_map[key].add(i)
                line_map[key].add(j)
        
        # Initial score and validity check
        current_score = 0
        for key, indices in line_map.items():
            if len(indices) == 4:
                current_score += 1
            elif len(indices) >= 5:
                return None, -1

        max_iter = 50000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, n - 1)
            old_point = current_points[idx]
            
            # Generate new point
            if random.random() < 0.5:
                new_x = old_point[0] + random.randint(-20, 20)
                new_y = old_point[1] + random.randint(-20, 20)
            else:
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            new_point = (new_x, new_y)
            
            # Check for duplicate point
            if new_point in current_points and new_point != old_point:
                continue

            # --- Incremental Update ---
            
            # 1. Identify lines affected by removing idx
            affected_keys = [key for key, indices in line_map.items() if idx in indices]
            
            # 2. Remove idx from these lines
            delta_score = 0
            for key in affected_keys:
                count = len(line_map[key])
                if count == 4:
                    delta_score -= 1
                line_map[key].discard(idx)
                if not line_map[key]:
                    del line_map[key]
            
            # 3. Determine lines formed by new_point
            new_lines = {}
            for j in range(n):
                if j == idx:
                    continue
                key = get_line_key(new_point, current_points[j])
                if key not in new_lines:
                    new_lines[key] = set()
                new_lines[key].add(j)
                
            # 4. Check validity (no 5 collinear)
            valid_move = True
            for key in new_lines:
                # Current count on this line (excluding idx, which is removed)
                current_count = len(line_map.get(key, set()))
                # New count will be current_count + 1
                if current_count + 1 >= 5:
                    valid_move = False
                    break
            
            if not valid_move:
                # Rollback
                for key in affected_keys:
                    if key not in line_map:
                        line_map[key] = set()
                    line_map[key].add(idx)
                continue
            
            # 5. Apply move
            for key in new_lines:
                count_before = len(line_map.get(key, set()))
                # count_after = count_before + 1
                if count_before == 3:
                    delta_score += 1
                
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(idx)
            
            current_points[idx] = new_point
            current_score += delta_score

        return current_points, current_score

    # Main search loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts % 5 == 0:
            init_pts = random_grid_init()
        else:
            init_pts = random_init()
            
        pts, score = hill_climb(init_pts)
        
        if score > best_score:
            best_score = score
            best_points = pts
            
        if best_score >= 15:
            break

    if best_points is None:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return [[p[0], p[1]] for p in best_points]
