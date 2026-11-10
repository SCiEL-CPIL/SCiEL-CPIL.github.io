# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    import math
    from itertools import combinations

    n = 16
    # Initial state: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_score(pts):
        """Returns (score, penalty) where penalty is number of lines with >= 5 points"""
        lines = {}
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                A = y2 - y1
                B = -(x2 - x1)
                C = x2*y1 - x1*y2
                
                g = math.gcd(A, math.gcd(B, C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    continue
                    
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                elif A == 0 and B == 0:
                    continue
                
                key = (A, B, C)
                
                if key not in lines:
                    lines[key] = 0
                lines[key] += 1
        
        score = 0
        penalty = 0
        
        for key, count in lines.items():
            disc = 1 + 8 * count
            sqrt_disc = int(math.isqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                continue
            k = (1 + sqrt_disc) // 2
            if k * (k-1) // 2 != count:
                continue
            
            if k >= 5:
                penalty += 1
            if k == 4:
                score += 1
                
        return score, penalty

    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    def egcd(a, b):
        if b == 0:
            return (1, 0, a)
        x1, y1, d = egcd(b, a % b)
        return (y1, x1 - (a // b) * y1, d)

    def get_line_points(pts, line_key):
        """Get indices of points on the given line"""
        A, B, C = line_key
        res = []
        for i, (x, y) in enumerate(pts):
            if A*x + B*y + C == 0:
                res.append(i)
        return res

    def try_complete_line(pts, target_line_key, exclude_idx):
        """
        Try to move a point (not exclude_idx) to a new position on target_line_key
        such that it doesn't collide with other points.
        Returns (new_index, new_pos) or (None, None)
        """
        A, B, C = target_line_key
        g = math.gcd(A, B)
        if g == 0: return None, None
        if -C % g != 0: return None, None
        
        x, y, d = egcd(A, B)
        xp = x * (-C // d)
        yp = y * (-C // d)
        
        step_x = B // g
        step_y = -A // g
        
        # Find points currently on this line
        existing_points = set()
        for i in range(n):
            if i == exclude_idx: continue
            px, py = pts[i]
            if A*px + B*py + C == 0:
                existing_points.add((px, py))
        
        # We want to add a point. 
        # If the line already has 4 points, adding a 5th is bad (penalty).
        # We should only try this if the line has < 4 points, or if we are forced.
        # But we are calling this to *create* a 4-point line, so ideally it has 3.
        
        # Try random t values
        for _ in range(50):
            t = random.randint(-100, 100)
            nx = xp + step_x * t
            ny = yp + step_y * t
            
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                if (nx, ny) not in existing_points:
                    # Check if it collides with the point we are moving?
                    # We are moving exclude_idx. So we must ensure (nx,ny) != pts[exclude_idx]
                    if pts[exclude_idx] != [nx, ny]:
                        return exclude_idx, [nx, ny]
        return None, None

    start_time = time.time()
    time_limit = 115.0
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_pts = [p[:] for p in points]
    current_score, current_penalty = get_score(current_pts)
    # Fitness = score - penalty * 10
    current_fitness = current_score - current_penalty * 10
    
    best_pts = [p[:] for p in points]
    best_score = current_score
    best_fitness = current_fitness
    
    while time.time() - start_time < time_limit:
        new_pts = [p[:] for p in current_pts]
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Targeted move: Complete a 3-point line
            # Find lines with exactly 3 points
            lines_dict = {}
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1 = current_pts[i]
                    x2, y2 = current_pts[j]
                    A = y2 - y1
                    B = -(x2 - x1)
                    C = x2*y1 - x1*y2
                    g = math.gcd(A, math.gcd(B, C))
                    if g != 0:
                        A //= g
                        B //= g
                        C //= g
                    else:
                        continue
                    if A < 0 or (A == 0 and B < 0):
                        A = -A
                        B = -B
                        C = -C
                    key = (A, B, C)
                    if key not in lines_dict:
                        lines_dict[key] = 0
                    lines_dict[key] += 1
            
            # Filter lines with count corresponding to k=3 (count=3)
            candidate_lines = []
            for key, count in lines_dict.items():
                disc = 1 + 8 * count
                sqrt_disc = int(math.isqrt(disc))
                if sqrt_disc * sqrt_disc == disc:
                    k = (1 + sqrt_disc) // 2
                    if k == 3:
                        candidate_lines.append(key)
            
            if candidate_lines:
                target_line = random.choice(candidate_lines)
                # Pick a point to move. Prefer one not on the line.
                points_on_line = set()
                A, B, C = target_line
                for i in range(n):
                    x, y = current_pts[i]
                    if A*x + B*y + C == 0:
                        points_on_line.add(i)
                
                # We want to move a point NOT on the line to the line
                # Or move a point ON the line to a different position on the line? No, that doesn't increase count.
                # So we must move a point from outside to inside.
                outside_points = [i for i in range(n) if i not in points_on_line]
                if outside_points:
                    k = random.choice(outside_points)
                    new_k, new_pos = try_complete_line(current_pts, target_line, k)
                    if new_pos:
                        new_pts[new_k] = new_pos
                    else:
                        # Fallback
                        new_pts[k] = random_point()
                else:
                    new_pts[random.randint(0, n-1)] = random_point()
            else:
                # No 3-point lines, do a random move
                k = random.randint(0, n-1)
                new_pts[k] = random_point()
                
        elif move_type < 0.7:
            # Small local move
            k = random.randint(0, n-1)
            new_pts[k] = [current_pts[k][0] + random.randint(-10, 10), 
                          current_pts[k][1] + random.randint(-10, 10)]
            
        else:
            # Random jump
            k = random.randint(0, n-1)
            new_pts[k] = random_point()
        
        new_pts[k][0] = max(-1000, min(1000, new_pts[k][0]))
        new_pts[k][1] = max(-1000, min(1000, new_pts[k][1]))
        
        new_score, new_penalty = get_score(new_pts)
        new_fitness = new_score - new_penalty * 10
        
        # Update best valid
        if new_penalty == 0 and new_score > best_score:
            best_score = new_score
            best_pts = [p[:] for p in new_pts]
            
        delta = new_fitness - current_fitness
        if delta > 0 or random.random() < math.exp(delta / T):
            current_pts = new_pts
            current_score = new_score
            current_penalty = new_penalty
            current_fitness = new_fitness
            
        T = max(T_min, T * cooling_rate)
        
    return best_pts
# EVOLVE-BLOCK-END
