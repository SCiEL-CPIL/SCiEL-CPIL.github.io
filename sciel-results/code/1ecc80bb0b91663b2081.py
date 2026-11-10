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
        """Returns (is_valid, score, lines_dict)"""
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
                    lines[key] = []
                lines[key].append(i)
                
                if len(lines[key]) == 2:
                    lines[key].append(j)
                # We need to be careful. The pair (i,j) adds both to the list?
                # No, the loop adds i then j?
                # The current code just counts. Let's change to store indices to identify 3/4/5 sets.
                
        # Re-implement to store indices properly
        lines_map = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                A = y2 - y1
                B = -(x2 - x1)
                C = x2*y1 - x1*y2
                
                g = math.gcd(A, math.gcd(B, C))
                if g == 0:
                    continue
                A //= g
                B //= g
                C //= g
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                key = (A, B, C)
                if key not in lines_map:
                    lines_map[key] = set()
                lines_map[key].add(i)
                lines_map[key].add(j)
        
        score = 0
        is_valid = True
        lines_3 = []
        lines_4 = []
        lines_5_plus = []
        
        for key, pts_on_line in lines_map.items():
            k = len(pts_on_line)
            if k >= 5:
                is_valid = False
                lines_5_plus.append(pts_on_line)
            elif k == 4:
                score += 1
                lines_4.append(pts_on_line)
            elif k == 3:
                lines_3.append(pts_on_line)
                
        return is_valid, score, lines_3, lines_5_plus

    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    def try_move_to_line(pts, k, i, j):
        """
        Try to move point k to a position on the line defined by points i and j.
        """
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        
        A = y2 - y1
        B = -(x2 - x1)
        C = x2*y1 - x1*y2
        
        g = math.gcd(A, B)
        if g == 0: return None
        if -C % g != 0: return None
        
        def egcd(a, b):
            if b == 0:
                return (1, 0, a)
            x1, y1, d = egcd(b, a % b)
            return (y1, x1 - (a // b) * y1, d)
        
        x, y, d = egcd(A, B)
        xp = x * (-C // d)
        yp = y * (-C // d)
        
        step_x = B // g
        step_y = -A // g
        
        # Try a few t values
        for _ in range(10):
            t = random.randint(-20, 20)
            nx = xp + step_x * t
            ny = yp + step_y * t
            
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                if [nx, ny] != pts[i] and [nx, ny] != pts[j] and [nx, ny] != pts[k]:
                    is_distinct = True
                    for l in range(n):
                        if l == k: continue
                        if pts[l] == [nx, ny]:
                            is_distinct = False
                            break
                    if is_distinct:
                        return [nx, ny]
        return None

    start_time = time.time()
    time_limit = 110.0
    
    current_pts = [p[:] for p in points]
    current_valid, current_score, _, _ = get_score(current_pts)
    best_pts = [p[:] for p in current_pts]
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        new_pts = [p[:] for p in current_pts]
        new_valid, new_score, lines_3, lines_5_plus = get_score(current_pts)
        
        moved = False
        
        # 1. Repair: If invalid, try to break a 5-collinear line
        if not new_valid and lines_5_plus:
            # Pick a random 5-collinear line
            line_pts = random.choice(lines_5_plus)
            # Pick a random point from that line to move
            k = random.choice(list(line_pts))
            # Move it to a random location
            new_pts[k] = random_point()
            moved = True
        # 2. Expand: If valid and there are 3-collinear lines, try to make them 4
        elif new_valid and lines_3:
            line_pts = random.choice(lines_3)
            # Pick a point NOT on the line
            candidates = [i for i in range(n) if i not in line_pts]
            if candidates:
                k = random.choice(candidates)
                # Pick 2 points from the line to define the line
                i, j = random.sample(list(line_pts), 2)
                new_pos = try_move_to_line(new_pts, k, i, j)
                if new_pos:
                    new_pts[k] = new_pos
                    moved = True
        
        # 3. Random Perturbation if no specific move was made or failed
        if not moved:
            k = random.randint(0, n-1)
            # Small random move
            new_pts[k] = [current_pts[k][0] + random.randint(-5, 5), 
                          current_pts[k][1] + random.randint(-5, 5)]
            new_pts[k][0] = max(-1000, min(1000, new_pts[k][0]))
            new_pts[k][1] = max(-1000, min(1000, new_pts[k][1]))
        
        # Evaluate new state
        new_valid, new_score, _, _ = get_score(new_pts)
        
        if new_valid:
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = [p[:] for p in current_pts]
        
        T = max(T_min, T * cooling_rate)
        
    return best_pts
# EVOLVE-BLOCK-END
