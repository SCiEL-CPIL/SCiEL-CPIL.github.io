import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1
    best_points = None
    
    # Helper to normalize line coefficients
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
        # Normalize sign: A > 0 or (A == 0 and B > 0) or (A == 0 and B == 0 and C > 0)
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0:
            if B < 0:
                A, B, C = -A, -B, -C
            elif B == 0:
                if C < 0:
                    A, B, C = -A, -B, -C
        return (A, B, C)

    def evaluate(points):
        # points is a list of (x, y)
        if len(points) != 16:
            return -1
        if len(set(points)) != 16:
            return -1
            
        # Check bounds
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return -1

        line_map = defaultdict(list)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                line_map[key].append(i)
        
        score = 0
        for key, indices in line_map.items():
            count = len(indices)
            if count >= 5:
                return -1 # Invalid: five collinear
            if count == 4:
                score += 1
        return score

    def random_grid_init():
        # 4x4 grid with some random offset and scaling to allow exploration
        # But we want integer points.
        # Let's just use a standard 4x4 grid shifted randomly
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
        current_score = evaluate(current_points)
        
        if current_score < 0:
             # Try to fix validity by random restarts within this hill climb?
             # Or just return -1
             pass

        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Generate a new position
            # Strategy: move to a random nearby point or a random point
            if random.random() < 0.5:
                new_x = old_x + random.randint(-20, 20)
                new_y = old_y + random.randint(-20, 20)
            else:
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            new_point = (new_x, new_y)
            
            # Check if new point collides with others
            if new_point in current_points and new_point != current_points[idx]:
                continue
                
            new_points = current_points.copy()
            new_points[idx] = new_point
            
            new_score = evaluate(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
            elif new_score == current_score and random.random() < 0.1:
                # Accept equal score to escape local optima (plateau walking)
                current_points = new_points
                # current_score remains same
                
        return current_points, current_score

    # Main search loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts % 10 == 0:
            init_pts = random_grid_init()
        else:
            init_pts = random_init()
            
        pts, score = hill_climb(init_pts)
        
        if score > best_score:
            best_score = score
            best_points = pts
            
        # If we found a very good score, we can try to refine it more
        if best_score >= 15:
            break

    if best_points is None:
        # Fallback to seed
        return [[x, y] for x in range(4) for y in range(4)]
        
    return [[p[0], p[1]] for p in best_points]
