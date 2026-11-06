import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        triples_with_idx = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
                        
        best_score = 0
        
        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            if current_size + (n - step) <= best_score:
                return
            idx = order[step]
            bbb(step + 1, current_mask, current_size)
            valid = True
            for a, b in triples_with_idx[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def greedy_construct(candidate_pool, target_size=20):
        """Greedy construction that maximizes triple density while avoiding 4-collinear."""
        current_set = []
        current_set_set = set()
        
        while len(current_set) < target_size:
            best_candidate = None
            best_score = -1
            
            for p in candidate_pool:
                if p in current_set_set:
                    continue
                
                valid = True
                triples_count = 0
                new_lines_count = 0
                
                for i in range(len(current_set)):
                    a = current_set[i]
                    for j in range(i + 1, len(current_set)):
                        b = current_set[j]
                        x1, y1 = a
                        x2, y2 = b
                        xp, yp = p
                        if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                            # Check if this creates a 4th collinear point
                            found_third = False
                            for k in range(len(current_set)):
                                if k == i or k == j:
                                    continue
                                c = current_set[k]
                                xc, yc = c
                                if (x2 - x1) * (yc - y1) == (y2 - y1) * (xc - x1):
                                    found_third = True
                                    break
                            if found_third:
                                valid = False
                                break
                            else:
                                triples_count += 1
                if not valid:
                    continue
                
                # Score: prefer more triples, but also consider line diversity
                score = triples_count * 10 + len(current_set) * 2
                if score > best_score:
                    best_score = score
                    best_candidate = p
            
            if best_candidate is None:
                break
                
            current_set.append(best_candidate)
            current_set_set.add(best_candidate)
            
        return current_set

    best_points = None
    best_val = float('inf')
    
    # Strategy 1: Greedy on various grids with more diverse parameters
    grid_configs = []
    for size in range(6, 16):
        for step in range(2, 12):
            if size * step > 250:
                continue
            for ox in range(-25, 26, 5):
                for oy in range(-25, 26, 5):
                    pts = []
                    for i in range(size):
                        for j in range(size):
                            x = ox + i * step
                            y = oy + j * step
                            if -100 <= x <= 100 and -100 <= y <= 100:
                                pts.append((x, y))
                    if len(pts) >= 20:
                        grid_configs.append(pts)
    
    # Shuffle and try a subset
    random.shuffle(grid_configs)
    max_grids = min(50, len(grid_configs))
    for idx, pts in enumerate(grid_configs[:max_grids]):
        if elapsed() > deadline - 40:
            break
        cand = greedy_construct(pts, 20)
        if len(cand) == 20:
            if has_4_collinear(cand):
                continue
            s = calculate_score(cand)
            if s < best_val:
                best_val = s
                best_points = [tuple(p) for p in cand]
    
    # Strategy 2: Multiple random initializations
    if elapsed() < deadline - 35:
        for trial in range(15):
            if elapsed() > deadline - 20:
                break
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                s = calculate_score(pts)
                if s < best_val:
                    best_val = s
                    best_points = [tuple(p) for p in pts]
    
    # Strategy 3: Intensive local search
    if best_points is not None:
        current_pts = list(best_points)
        current_val = best_val
        
        # Precompute triples for faster checking
        def check_move_validity(current_pts, idx, new_p):
            """Check if placing new_p at idx creates 4-collinear."""
            xp, yp = new_p
            n = len(current_pts)
            for i in range(n):
                if i == idx:
                    continue
                x1, y1 = current_pts[i]
                for j in range(i + 1, n):
                    if j == idx:
                        continue
                    x2, y2 = current_pts[j]
                    # Check if new_p is on line through i,j
                    if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                        # Check if there's already a third point on this line
                        for k in range(n):
                            if k == idx or k == i or k == j:
                                continue
                            x3, y3 = current_pts[k]
                            if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                                return False
            return True
        
        for iteration in range(500):
            if elapsed() > deadline - 10:
                break
                
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # Diverse move generation
            r = random.random()
            if r < 0.3:
                # Small local move
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
            elif r < 0.6:
                # Medium local move
                new_p = (old_p[0] + random.randint(-15, 15), old_p[1] + random.randint(-15, 15))
            else:
                # Random global move
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                
            if new_p in current_pts:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
                
            if not check_move_validity(current_pts, idx, new_p):
                continue
                
            new_pts = current_pts[:]
            new_pts[idx] = new_p
            
            s = calculate_score(new_pts)
            if s < current_val:
                current_val = s
                current_pts = new_pts
                best_val = s
                best_points = current_pts
            elif s == current_val and random.random() < 0.03:
                if elapsed() < deadline - 25:
                    current_pts = new_pts
                    best_points = current_pts

    # Fallback
    if best_points is None:
        while True:
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break

    return [[x, y] for x, y in best_points]
