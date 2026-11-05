import time
import random
import math

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = _gcd(dx, dy)
    if g == 0:
        return (0, 0, p1[0], p1[1])
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * p1[0] - dx * p1[1]
    return (dx, dy, c)

def _collinear3(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _count_triples_and_check_four(points):
    n = len(points)
    line_to_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    
    triples = 0
    has_four = False
    for key, pts in line_to_pts.items():
        c = len(pts)
        if c >= 4:
            has_four = True
        if c >= 3:
            triples += c * (c - 1) * (c - 2) // 6
    return triples, has_four

def _has_four_collinear(points):
    _, has_four = _count_triples_and_check_four(points)
    return has_four

def _count_collinear_triples(points):
    triples, _ = _count_triples_and_check_four(points)
    return triples

def _max_3free_subset(points, deadline=None):
    n = len(points)
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mask = 0
            for k in range(n):
                if i == k or j == k:
                    continue
                if _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask

    best = [0]
    
    def can_add(i, current_mask):
        temp = current_mask
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            temp -= lsb
            if (collinear_mask[i][j] & current_mask) != 0:
                return False
        return True

    def dfs(current_mask, candidates_mask, current_size):
        if deadline and time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
        if candidates_mask == 0:
            return
        if current_size + bin(candidates_mask).count('1') <= best[0]:
            return
        lsb = candidates_mask & -candidates_mask
        i = lsb.bit_length() - 1
        dfs(current_mask, candidates_mask & ~lsb, current_size)
        if can_add(i, current_mask):
            new_current = current_mask | lsb
            dfs(new_current, candidates_mask & ~lsb, current_size + 1)

    dfs(0, (1 << n) - 1, 0)
    return best[0]

def _try_line_packing(deadline=None):
    """Systematically build points to create 3-point lines."""
    points = []
    seen = set()
    
    base_x, base_y = random.randint(-50, 50), random.randint(-50, 50)
    dx, dy = random.choice([(2,0),(0,2),(1,1),(1,-1),(2,1),(1,2)])
    p0 = [base_x, base_y]
    p1 = [base_x + dx, base_y + dy]
    p2 = [base_x + 2*dx, base_y + 2*dy]
    for p in [p0, p1, p2]:
        if abs(p[0]) <= 100 and abs(p[1]) <= 100 and tuple(p) not in seen:
            points.append(p)
            seen.add(tuple(p))
    
    while len(points) < 20:
        if deadline and time.time() > deadline:
            break
        best_candidate = None
        best_gain = -1
        
        for _ in range(50):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            cand = [x, y]
            if tuple(cand) in seen:
                continue
            if abs(x) > 100 or abs(y) > 100:
                continue
            
            test_points = points + [cand]
            if _has_four_collinear(test_points):
                continue
            
            new_triples = 0
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if _collinear3(points[i], points[j], cand):
                        new_triples += 1
            
            if new_triples > best_gain:
                best_gain = new_triples
                best_candidate = cand
        
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for t in [-2, -1, 1, 2, 3]:
                    if t == 0: continue
                    nx = x1 + t * dx
                    ny = y1 + t * dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    cand = [nx, ny]
                    if tuple(cand) in seen:
                        continue
                    test_points = points + [cand]
                    if _has_four_collinear(test_points):
                        continue
                    new_triples = 0
                    for a in range(len(points)):
                        for b in range(a+1, len(points)):
                            if _collinear3(points[a], points[b], cand):
                                new_triples += 1
                    if new_triples > best_gain:
                        best_gain = new_triples
                        best_candidate = cand
        
        if best_candidate:
            points.append(best_candidate)
            seen.add(tuple(best_candidate))
        else:
            for _ in range(100):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if tuple(cand) in seen or abs(x) > 100 or abs(y) > 100:
                    continue
                test_points = points + [cand]
                if _has_four_collinear(test_points):
                    continue
                points.append(cand)
                seen.add(tuple(cand))
                break
            else:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                points.append(cand)
                seen.add(tuple(cand))
    
    return points[:20]

def _local_search_line_aware(points, deadline):
    """Local search that specifically tries to create new collinear triples."""
    current = [p[:] for p in points]
    current_triples = _count_collinear_triples(current)
    best = [p[:] for p in current]
    best_triples = current_triples
    
    while time.time() < deadline:
        idx = random.randint(0, 19)
        orig = current[idx][:]
        
        moved = False
        for _ in range(30):
            i = random.randint(0, 19)
            j = random.randint(0, 19)
            if i == j or i == idx or j == idx:
                continue
            x1, y1 = current[i]
            x2, y2 = current[j]
            dx = x2 - x1
            dy = y2 - y1
            for t in random.sample([-3,-2,-1,1,2,3], 2):
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                cand = [nx, ny]
                if any(tuple(cand) == tuple(p) for k, p in enumerate(current) if k != idx):
                    continue
                test = current[:]
                test[idx] = cand
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                if new_triples > current_triples:
                    current[idx] = cand
                    current_triples = new_triples
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best = [p[:] for p in current]
                    moved = True
                    break
            if moved:
                break
        
        if not moved:
            for _ in range(5):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if any(tuple(cand) == tuple(p) for k, p in enumerate(current) if k != idx):
                    continue
                test = current[:]
                test[idx] = cand
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                if new_triples > current_triples:
                    current[idx] = cand
                    current_triples = new_triples
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best = [p[:] for p in current]
                    moved = True
                    break
        
        if not moved:
            current[idx] = orig
    
    return best

def _directed_moves(points):
    """Generate directed moves that try to create new collinear triples."""
    n = len(points)
    moves = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = _gcd(dx, dy)
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            
            for t in range(1, 4):
                for sign in [1, -1]:
                    new_p = [p2[0] + sign * t * dx_n, p2[1] + sign * t * dy_n]
                    if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                        if new_p not in points:
                            moves.append(new_p)
                    new_p = [p1[0] - sign * t * dx_n, p1[1] - sign * t * dy_n]
                    if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                        if new_p not in points:
                            moves.append(new_p)
    
    for _ in range(10):
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        if new_p not in points:
            moves.append(new_p)
    
    return moves

def _directed_local_search(points, deadline):
    """Local search using directed moves that target collinear triples."""
    current = [p[:] for p in points]
    current_triples = _count_collinear_triples(current)
    best = [p[:] for p in current]
    best_triples = current_triples
    
    while time.time() < deadline:
        all_moves = _directed_moves(current)
        random.shuffle(all_moves)
        
        improved = False
        for new_p in all_moves:
            if time.time() > deadline:
                break
            
            best_move = None
            best_move_score = current_triples
            
            for i in range(20):
                if current[i] == new_p:
                    continue
                test = current[:]
                test[i] = new_p
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                if new_triples > best_move_score:
                    best_move_score = new_triples
                    best_move = i
            
            if best_move is not None:
                current[best_move] = new_p
                current_triples = best_move_score
                improved = True
                if current_triples > best_triples:
                    best_triples = current_triples
                    best = [p[:] for p in current]
                break
        
        if not improved:
            # Random perturbation
            idx = random.randint(0, 19)
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if any(tuple(cand) == tuple(p) for k, p in enumerate(current) if k != idx):
                    continue
                test = current[:]
                test[idx] = cand
                if _has_four_collinear(test):
                    continue
                new_triples = _count_collinear_triples(test)
                if new_triples > current_triples:
                    current[idx] = cand
                    current_triples = new_triples
                    if current_triples > best_triples:
                        best_triples = current_triples
                        best = [p[:] for p in current]
                    break
    
    return best

def _grid_based_initial():
    """Generate initial points from a grid structure with many collinearities."""
    pts = []
    for x in range(4):
        for y in range(5):
            pts.append([x * 10, y * 10])
    
    current_pts = [p[:] for p in pts]
    attempts = 0
    while attempts < 500:
        _, has_four = _count_triples_and_check_four(current_pts)
        if not has_four:
            return current_pts
        
        line_to_points = {}
        for i in range(20):
            for j in range(i + 1, 20):
                key = _line_key(current_pts[i], current_pts[j])
                if key not in line_to_points:
                    line_to_points[key] = set()
                line_to_points[key].add(i)
                line_to_points[key].add(j)
        
        moved = False
        for key, pts_set in line_to_points.items():
            if len(pts_set) >= 4:
                idx = random.choice(list(pts_set))
                old_p = current_pts[idx]
                for _ in range(50):
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    if dx == 0 and dy == 0:
                        continue
                    new_p = [old_p[0] + dx, old_p[1] + dy]
                    if -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                        if new_p not in current_pts:
                            current_pts[idx] = new_p
                            moved = True
                            break
                if moved:
                    break
        attempts += 1
    
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_points = None
    best_score = 20
    
    # Try multiple line-packing constructions
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        pts = _try_line_packing(deadline - 5)
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        # Do local search to improve
        improved = _local_search_line_aware(pts, deadline - 5)
        
        # Evaluate
        score = _max_3free_subset(improved, deadline - 2)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]
    
    # Try directed local search
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        pts = _try_line_packing(deadline - 5)
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        improved = _directed_local_search(pts, deadline - 5)
        
        score = _max_3free_subset(improved, deadline - 2)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]
    
    # Try grid-based initial
    for attempt in range(3):
        if time.time() > deadline - 10:
            break
        pts = _grid_based_initial()
        if len(pts) < 20 or len(set(tuple(p) for p in pts)) != 20:
            continue
        if _has_four_collinear(pts):
            continue
        
        improved = _directed_local_search(pts, deadline - 5)
        
        score = _max_3free_subset(improved, deadline - 2)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in improved]
    
    if best_points is None:
        # Fallback: random construction
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if tuple(p) in seen:
                continue
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
                continue
            seen.add(tuple(p))
        best_points = points
    
    return best_points
