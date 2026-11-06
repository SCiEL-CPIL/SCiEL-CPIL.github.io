def construct():
    """
    Find 16 distinct integer points with |x|, |y| <= 1000, no five collinear,
    maximizing the number of lines containing exactly four of the points.
    
    Uses an improved simulated annealing with structured move generation.
    """
    random.seed(42)
    np.random.seed(42)
    
    def get_lines_and_scores(points):
        """
        Returns (line_points_dict, count_4_lines, has_5_collinear)
        """
        line_points = defaultdict(set)
        n = len(points)
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                # Normalize direction
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                # Line coefficients a, b, c for ax + by + c = 0
                a = dy_n
                b = -dx_n
                c = -(a * x1 + b * y1)
                
                # Normalize (a, b, c)
                g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                
                if a < 0 or (a == 0 and b < 0):
                    a, b, c = -a, -b, -c
                
                line_key = (a, b, c)
                line_points[line_key].add(i)
        
        count_4 = 0
        has_5 = False
        for key, pts in line_points.items():
            l = len(pts)
            if l == 4:
                count_4 += 1
            elif l >= 5:
                has_5 = True
        
        return line_points, count_4, has_5

    def are_valid(points):
        """Check distinctness and bounds."""
        if len(set(tuple(p) for p in points)) != 16:
            return False
        for x, y in points:
            if abs(x) > 1000 or abs(y) > 1000:
                return False
        return True

    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    # Initial configuration: 4x4 grid
    # Points: (0,0) to (3,3) shifted to center
    initial_points = [[x, y] for x in [-2, -1, 0, 1] for y in [-2, -1, 0, 1]]
    
    current = [p[:] for p in initial_points]
    _, current_score, has_5 = get_lines_and_scores(current)
    
    if has_5:
        # Fallback to random if grid is bad
        while True:
            pts = [random_point() for _ in range(16)]
            if are_valid(pts):
                _, s, h5 = get_lines_and_scores(pts)
                if not h5:
                    current = pts
                    current_score = s
                    break
    else:
        current_score = current_score

    best = [p[:] for p in current]
    best_score = current_score

    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    max_iter = 200000 # Increased iterations

    for it in range(max_iter):
        if T < T_min:
            break
            
        new_points = [p[:] for p in current]
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.3:
            # Type 1: Small local perturbation
            idx = random.randint(0, 15)
            delta_x = random.randint(-5, 5)
            delta_y = random.randint(-5, 5)
            new_points[idx][0] += delta_x
            new_points[idx][1] += delta_y
            
        elif move_type < 0.7:
            # Type 2: Snap to a line defined by 3 other points
            idx = random.randint(0, 15)
            other_indices = [i for i in range(16) if i != idx]
            # Pick 3 random other points
            triplet = random.sample(other_indices, 3)
            p1, p2, p3 = current[triplet[0]], current[triplet[1]], current[triplet[2]]
            
            # Check if p1, p2, p3 are collinear
            if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                # They are collinear. Find the line equation.
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                
                a = dy_n
                b = -dx_n
                c = -(a * p1[0] + b * p1[1])
                
                # Normalize
                g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0):
                    a, b, c = -a, -b, -c
                
                # We want to place new_points[idx] on this line.
                # The line is ax + by + c = 0.
                # We need an integer solution (x,y) close to current[idx] or just any valid one.
                
                cx, cy = current[idx]
                found = False
                if b != 0:
                    # Try x near cx
                    for dx_try in range(-20, 21):
                        x_try = cx + dx_try
                        if abs(x_try) > 1000:
                            continue
                        num = -c - a * x_try
                        if num % b == 0:
                            y_try = num // b
                            if abs(y_try) <= 1000:
                                # Check if this point is already taken by another index
                                candidate = [x_try, y_try]
                                is_taken = False
                                for k in range(16):
                                    if k != idx and current[k] == candidate:
                                        is_taken = True
                                        break
                                if not is_taken:
                                    new_points[idx] = candidate
                                    found = True
                                    break
                elif a != 0:
                    # Vertical line x = -c/a
                    if -c % a == 0:
                        x_try = -c // a
                        if abs(x_try) <= 1000:
                            # Pick a random y? No, it must be on the line, which is vertical.
                            # Any y works for the line equation, but we need to be careful about collisions.
                            for dy_try in range(-20, 21):
                                y_try = cy + dy_try
                                if abs(y_try) <= 1000:
                                    candidate = [x_try, y_try]
                                    is_taken = False
                                    for k in range(16):
                                        if k != idx and current[k] == candidate:
                                            is_taken = True
                                            break
                                    if not is_taken:
                                        new_points[idx] = candidate
                                        found = True
                                        break
                else:
                    # Degenerate, shouldn't happen
                    pass
                
                if not found:
                    # Fallback to small perturbation if snap failed
                    delta_x = random.randint(-5, 5)
                    delta_y = random.randint(-5, 5)
                    new_points[idx][0] += delta_x
                    new_points[idx][1] += delta_y
            else:
                # Not collinear, fallback to small perturbation
                delta_x = random.randint(-5, 5)
                delta_y = random.randint(-5, 5)
                new_points[idx][0] += delta_x
                new_points[idx][1] += delta_y
                
        else:
            # Type 3: Random jump (escape local minima)
            idx = random.randint(0, 15)
            new_points[idx] = random_point()

        if not are_valid(new_points):
            T *= cooling
            continue
            
        _, new_score, new_has_5 = get_lines_and_scores(new_points)
        
        if new_has_5:
            # Reject if 5 collinear
            T *= cooling
            continue

        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = [p[:] for p in current]
        
        T *= cooling

    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
