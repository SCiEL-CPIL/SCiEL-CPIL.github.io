def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    best_score = float('inf')
    best_points = None
    
    def try_candidate(pts):
        nonlocal best_score, best_points
        if len(pts) != 12:
            return
        if len(set(map(tuple, pts))) != 12:
            return
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            return
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
    
    # Candidate 1: 3x4 grid
    try_candidate([[x, y] for x in range(4) for y in range(3)])
    
    # Candidate 2: Points on two perpendicular lines
    try_candidate([(i, 0) for i in range(6)] + [(0, i) for i in range(1, 7)])
    
    # Candidate 3: Three concurrent lines with 4 points each (sharing center)
    # Line 1: y=0, points at x=1,2,3,4
    # Line 2: x=0, points at y=1,2,3,4  
    # Line 3: y=x, points at (1,1),(2,2),(3,3),(4,4)
    # Center: (0,0)
    # Total: 1 + 4 + 4 + 4 = 13. Remove one.
    # Remove (4,4) to get 12.
    pts3 = [(0,0)] + [(i,0) for i in range(1,5)] + [(0,i) for i in range(1,5)] + [(i,i) for i in range(1,4)]
    try_candidate(pts3)
    
    # Candidate 4: 4 concurrent lines with 3 points each (sharing center)
    # Lines: y=0, x=0, y=x, y=-x
    # Each has 2 non-center points
    # Total: 1 + 4*2 = 9. Need 3 more.
    # Add 3 more points strategically.
    pts4 = [(0,0)] + [(i,0) for i in [1,-1]] + [(0,i) for i in [1,-1]] + [(i,i) for i in [1,-1]] + [(i,-i) for i in [1,-1]]
    # That's 9 points. Add 3 more.
    pts4 += [(2,1), (1,2), (-1,-2)]
    try_candidate(pts4)
    
    # Candidate 5: Regular hexagon-like with center
    import math
    pts5 = [(0,0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = round(3 * math.cos(angle))
        y = round(3 * math.sin(angle))
        pts5.append((x, y))
    # That's 7 points. Add 5 more.
    for i in range(5):
        angle = i * math.pi / 3 + math.pi/6
        x = round(5 * math.cos(angle))
        y = round(5 * math.sin(angle))
        pts5.append((x, y))
    # Check for duplicates
    if len(set(pts5)) == 12:
        try_candidate(pts5)
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)
    
    # Local search
    current_points = [list(p) for p in best_points]
    current_score = best_score
    
    # Try multiple restarts
    for restart in range(3):
        if time.time() > deadline:
            break
        
        if restart > 0:
            # Perturb from best
            current_points = [list(p) for p in best_points]
            for _ in range(3):
                idx = random.randint(0, 11)
                current_points[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
            # Ensure distinct
            while len(set(map(tuple, current_points))) < 12:
                idx = random.randint(0, 11)
                current_points[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
            current_score = count_ordinary_lines(current_points)
        else:
            current_points = [list(p) for p in best_points]
            current_score = best_score
        
        # Simulated annealing-like local search
        temp = 2.0
        for i in range(5000):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 11)
            old_x, old_y = current_points[idx]
            
            # Try to move to a position that creates collinearity
            # Pick 2 other points and try to put idx on their line
            if random.random() < 0.5:
                # Create new collinearity
                other_idx = random.sample([k for k in range(12) if k != idx], 2)
                x1, y1 = current_points[other_idx[0]]
                x2, y2 = current_points[other_idx[1]]
                # Extend the line
                dx = x2 - x1
                dy = y2 - y1
                # Pick a new point on this line
                t = random.randint(-5, 5)
                new_x = x1 + t * dx
                new_y = y1 + t * dy
            else:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            
            if (new_x, new_y) == (old_x, old_y):
                continue
            if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(current_points) if j != idx):
                continue
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
            
            current_points[idx] = [new_x, new_y]
            new_score = count_ordinary_lines(current_points)
            
            if new_score < current_score or random.random() < temp:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [list(p) for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
            
            temp *= 0.995
    
    return best_points
