import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_points = None
    best_score = -1
    
    # Initial candidates
    candidates = []
    
    # 1. Grid-like (Seed)
    grid = [[x, y] for x in range(4) for y in range(4)]
    candidates.append(grid)
    
    # Shifted grids
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            g = [[x+dx, y+dy] for x in range(4) for y in range(4)]
            candidates.append(g)

    # 2. Random
    for _ in range(20):
        pts = []
        while len(pts) < 16:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x,y] not in pts:
                pts.append([x,y])
        candidates.append(pts)

    # 3. Structured Random (Clusters on lines)
    for _ in range(20):
        pts = []
        success = True
        for _ in range(4):
            # Pick a random line
            x1 = random.randint(-100, 100)
            y1 = random.randint(-100, 100)
            x2 = random.randint(-100, 100)
            y2 = random.randint(-100, 100)
            if x1 == x2 and y1 == y2:
                success = False
                break
            
            # Generate 4 points on this line
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                success = False
                break
            step_x = dx // g
            step_y = dy // g
            
            # Points: (x1 + k*step_x, y1 + k*step_y)
            # Pick 4 distinct k values
            ks = random.sample(range(-10, 11), 4)
            for k in ks:
                px = x1 + k * step_x
                py = y1 + k * step_y
                if -1000 <= px <= 1000 and -1000 <= py <= 1000:
                    if [px, py] not in pts:
                        pts.append([px, py])
                    else:
                        success = False
                        break
            if not success:
                break
        if success and len(pts) == 16:
            candidates.append(pts)

    # Local Search
    for init_pts in candidates:
        if time.time() > deadline:
            break
            
        current_pts = [p[:] for p in init_pts]
        current_score, current_4, current_5 = evaluate(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_pts]
            
        # Hill Climbing
        no_improvement = 0
        max_iters = 20000 # Increase iterations per restart
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Choose move type
            # 70% Targeted, 30% Random
            if random.random() < 0.7:
                # Targeted: Find a 3-point line, try to extend
                move = generate_targeted_move(current_pts)
            else:
                # Random: Move a random point to a random location
                move = generate_random_move(current_pts)
            
            if move is None:
                continue
                
            idx, new_pos = move
            old_pos = current_pts[idx]
            current_pts[idx] = new_pos
            
            new_score, new_4, new_5 = evaluate(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                no_improvement = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_pts]
            else:
                # Revert
                current_pts[idx] = old_pos
                no_improvement += 1
                if no_improvement > 500:
                    break
                    
    if best_points is None:
        # Fallback to grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

def evaluate(points):
    n = len(points)
    if n != 16:
        return -1000, 0, 100
        
    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0: continue
            A //= g
            B //= g
            C //= g
            
            # Sign normalization
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    num_4 = 0
    num_5_plus = 0
    
    for key, pts_on_line in lines.items():
        count = len(pts_on_line)
        if count == 4:
            num_4 += 1
        elif count >= 5:
            num_5_plus += 1
            
    score = num_4 - 100 * num_5_plus
    return score, num_4, num_5_plus

def generate_targeted_move(points):
    n = len(points)
    lines = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0: continue
            A //= g; B //= g; C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A; B = -B; C = -C
            key = (A, B, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    three_point_lines = []
    for key, pts in lines.items():
        if len(pts) == 3:
            three_point_lines.append((key, list(pts)))
            
    if not three_point_lines:
        return generate_random_move(points)
        
    # Pick a random 3-point line
    key, line_pts = random.choice(three_point_lines)
    A, B, C = key
    
    # Pick a random point NOT on this line to move
    available = [i for i in range(n) if i not in line_pts]
    if not available:
        return generate_random_move(points)
        
    idx_to_move = random.choice(available)
    
    # Find a new position on this line
    new_pos = None
    for _ in range(50):
        if B != 0:
            x = random.randint(-1000, 1000)
            num = -A*x - C
            if num % B == 0:
                y = num // B
                if -1000 <= y <= 1000:
                    if [x, y] not in points:
                        new_pos = [x, y]
                        break
        elif A != 0:
            y = random.randint(-1000, 1000)
            num = -B*y - C
            if num % A == 0:
                x = num // A
                if -1000 <= x <= 1000:
                    if [x, y] not in points:
                        new_pos = [x, y]
                        break
        else:
            pass
            
    if new_pos is None:
        return generate_random_move(points)
        
    return idx_to_move, new_pos

def generate_random_move(points):
    idx = random.randint(0, 15)
    x = random.randint(-1000, 1000)
    y = random.randint(-1000, 1000)
    # Ensure distinct
    while [x, y] in points:
        x = random.randint(-1000, 1000)
        y = random.randint(-1000, 1000)
    return idx, [x, y]
